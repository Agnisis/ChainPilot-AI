import math
from typing import Any
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.config import settings
from app.services.features import create_time_series_features
from app.services.validation import walk_forward_validation, compute_extended_metrics, compute_prediction_intervals, adf_stationarity_test, ljung_box_residual_test
from app.services.ensemble import build_stacking_ensemble, weighted_average_ensemble
from app.services.hyperparameter_tuning import tune_model
from app.services.explainability import compute_shap_values
from app.services.deep_learning import HAS_TORCH, train_deep_model, forecast_future_deep, LSTMForecaster, GRUForecaster

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import pmdarima as pm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels/pmdarima not installed")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def temporal_train_test_split(df: pd.DataFrame, test_size: int = settings.test_size_days) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 60:
        test_size = max(7, int(len(df) * 0.2))
    test_size = min(test_size, max(7, int(len(df) * 0.3)))
    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    return train, test


def train_sarima(train_series: pd.Series, test_len: int, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)):
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=100)
    return fitted, fitted.forecast(steps=test_len)


def run_forecasting_pipeline(forecasting_df: pd.DataFrame) -> dict[str, Any]:
    feature_df = create_time_series_features(forecasting_df)
    model_df = feature_df.dropna().reset_index(drop=True)
    feature_cols = [
        c for c in model_df.columns if c not in ["Date", "Demand"] and pd.api.types.is_numeric_dtype(model_df[c])
    ]

    train_df, test_df = temporal_train_test_split(model_df)
    X_train, y_train = train_df[feature_cols], train_df["Demand"]
    X_test, y_test = test_df[feature_cols], test_df["Demand"]

    model_results: list[dict[str, Any]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    model_artifacts: dict[str, dict[str, Any]] = {}
    tuning_results = []
    base_train_preds = {}
    base_test_preds = {}

    def register(name, y_true, y_pred, dates, model_object=None, model_type="unknown", metadata=None):
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.clip(np.asarray(y_pred, dtype=float).reshape(-1), 0, None)
        min_len = min(len(y_true), len(y_pred), len(dates))
        
        metrics = compute_extended_metrics(y_true[:min_len], y_pred[:min_len])
        model_results.append({"Model": name, **metrics})
        
        prediction_frames[name] = pd.DataFrame(
            {
                "Date": pd.to_datetime(pd.Series(dates).iloc[:min_len].values),
                "Actual": y_true[:min_len],
                "Predicted": y_pred[:min_len],
            }
        )
        model_artifacts[name] = {
            "model": model_object,
            "type": model_type,
            "metadata": metadata or {},
            "metrics": metrics,
        }
        return y_pred[:min_len]

    # --- 1. Statistical Models ---
    if STATSMODELS_AVAILABLE:
        try:
            # Auto-ARIMA
            if not settings.fast_mode:
                auto_model = pm.auto_arima(y_train, seasonal=True, m=7, suppress_warnings=True, error_action="ignore")
                order = auto_model.order
                seasonal_order = auto_model.seasonal_order
            else:
                order, seasonal_order = (2, 1, 2), (0,0,0,0)
                
            fitted, preds = train_sarima(y_train, len(test_df), order=order, seasonal_order=seasonal_order)
            register("Auto-ARIMA", y_test.values, preds, test_df["Date"].values, fitted, "statistical", {"order": order, "seasonal": seasonal_order})
        except Exception as e:
            logger.warning(f"Auto-ARIMA failed: {e}")

        try:
            fitted, preds = train_sarima(y_train, len(test_df), order=(1,1,1), seasonal_order=(1,0,1,7))
            register("SARIMA", y_test.values, preds, test_df["Date"].values, fitted, "statistical", {"seasonal": True})
        except Exception as e:
            logger.warning(f"SARIMA failed: {e}")

    # --- 2. Machine Learning Models ---
    ml_models: dict[str, Any] = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200 if settings.fast_mode else 300,
            max_depth=16,
            min_samples_leaf=2,
            random_state=settings.random_state,
            n_jobs=-1,
        )
    }
    if XGBOOST_AVAILABLE:
        ml_models["XGBoost"] = XGBRegressor(n_estimators=300, learning_rate=0.04, max_depth=5, random_state=settings.random_state, n_jobs=-1)
    if LIGHTGBM_AVAILABLE:
        ml_models["LightGBM"] = LGBMRegressor(n_estimators=400, learning_rate=0.03, random_state=settings.random_state, verbose=-1)

    for model_name, model in ml_models.items():
        try:
            # Tune if enabled
            best_params = None
            if settings.tuning_mode and not settings.fast_mode:
                t_res = tune_model(model_name, X_train, y_train, X_test, y_test, n_trials=settings.tuning_trials)
                if t_res:
                    model.set_params(**t_res['best_params'])
                    tuning_results.append({
                        "model": model_name,
                        "best_params": t_res['best_params'],
                        "best_score": t_res['best_score'],
                        "n_trials": t_res['n_trials']
                    })

            model.fit(X_train, y_train)
            preds_test = model.predict(X_test)
            preds_train = model.predict(X_train)
            
            test_res = register(
                model_name, y_test.values, preds_test, test_df["Date"].values, 
                model, "machine_learning", {"feature_cols": feature_cols}
            )
            
            base_train_preds[model_name] = preds_train
            base_test_preds[model_name] = test_res
        except Exception as e:
            logger.warning(f"{model_name} failed: {e}")

    # --- 3. Deep Learning Models ---
    if HAS_TORCH and settings.deep_learning_enabled:
        dl_configs = [
            ("LSTM", LSTMForecaster),
            ("GRU", GRUForecaster)
        ]
        for name, cls in dl_configs:
            try:
                res = train_deep_model(
                    cls, X_train, y_train, X_test, y_test,
                    sequence_length=settings.dl_sequence_length,
                    hidden_size=settings.dl_hidden_size,
                    num_layers=settings.dl_num_layers,
                    epochs=settings.dl_epochs,
                    batch_size=settings.dl_batch_size,
                    patience=settings.dl_patience
                )
                if res:
                    preds = res['predictions']
                    # Align lengths (test_df vs preds)
                    min_l = min(len(preds), len(test_df))
                    register(
                        name, y_test.values[-min_l:], preds[-min_l:], test_df["Date"].values[-min_l:],
                        res['model_state'], "deep_learning", {"config": res['config'], "scaler": res['scaler']}
                    )
            except Exception as e:
                logger.warning(f"{name} failed: {e}")

    if not model_results:
        raise ValueError("No forecasting models trained successfully.")

    # --- 4. Ensemble (Stacking / Weighted) ---
    ensemble_weights = []
    if settings.ensemble_enabled and len(base_train_preds) >= 2:
        try:
            # Weighted average
            metrics_dict = {m['Model']: m['RMSE'] for m in model_results if m['Model'] in base_test_preds}
            w_res = weighted_average_ensemble(base_test_preds, metrics_dict, "RMSE")
            w_preds = w_res["predictions"]
            
            register("Weighted Ensemble", y_test.values, w_preds, test_df["Date"].values, None, "ensemble", {"weights": w_res["weights"]})
            
            # Stacking (Ridge)
            s_res = build_stacking_ensemble(base_train_preds, y_train.values, base_test_preds)
            s_preds = s_res["predictions"]
            register("Stacking Ensemble", y_test.values, s_preds, test_df["Date"].values, s_res["meta_model"], "ensemble", {"weights": s_res["weights"]})
            
            for m, w in s_res["weights"].items():
                ensemble_weights.append({"model": m, "weight": float(w)})
        except Exception as e:
            logger.warning(f"Ensemble failed: {e}")

    # --- 5. Selection & Advanced Diagnostics ---
    metrics_df = pd.DataFrame(model_results).sort_values("RMSE").reset_index(drop=True)
    metrics_df["Rank"] = np.arange(1, len(metrics_df) + 1)
    
    # Filter out ensembles from being the 'best model' for future forecasting logic simplicity
    base_metrics = metrics_df[~metrics_df['Model'].str.contains("Ensemble")]
    best_name = base_metrics.iloc[0]["Model"]
    best_artifact = model_artifacts[best_name]

    # Walk-forward CV for best model
    cv_scores = None
    if best_artifact["type"] == "machine_learning" and not settings.fast_mode:
        def factory():
            m = _new_ml_model(best_name)
            if best_artifact.get('model'):
                m.set_params(**best_artifact['model'].get_params())
            return m
            
        cv_res = walk_forward_validation(factory, model_df, feature_cols, "Demand", n_splits=settings.cv_folds)
        if cv_res:
            cv_scores = {best_name: cv_res['fold_results']}

    # Explainability (SHAP)
    feature_importance = None
    if best_artifact["type"] == "machine_learning":
        shap_res = compute_shap_values(best_artifact["model"], X_test)
        if shap_res:
            feature_importance = shap_res["features"]

    # Statistical Tests
    stat_tests = []
    adf_res = adf_stationarity_test(model_df["Demand"])
    if adf_res: stat_tests.append(adf_res)
    
    # Residuals & Prediction Intervals
    best_pred = prediction_frames[best_name]
    residuals = best_pred["Actual"] - best_pred["Predicted"]
    
    lb_res = ljung_box_residual_test(residuals.values)
    if lb_res: stat_tests.append(lb_res)
    
    intervals = compute_prediction_intervals(best_pred["Predicted"].values, residuals.values, [0.90, 0.95])
    conf_intervals = []
    if intervals:
        for i, dt in enumerate(best_pred["Date"]):
            conf_intervals.append({
                "date": dt.strftime("%Y-%m-%d"),
                "forecast": float(best_pred["Predicted"].iloc[i]),
                "lower_90": float(intervals["90"]["lower"][i]),
                "upper_90": float(intervals["90"]["upper"][i]),
                "lower_95": float(intervals["95"]["lower"][i]),
                "upper_95": float(intervals["95"]["upper"][i]),
            })

    # Future forecast
    future_df = _forecast_future(
        best_name, best_artifact, forecasting_df, feature_cols, horizon=max(settings.forecast_horizons)
    )

    return {
        "metrics_df": metrics_df,
        "prediction_frames": prediction_frames,
        "model_artifacts": model_artifacts,
        "best_model_name": best_name,
        "best_metrics": model_artifacts[best_name]["metrics"],
        "feature_df": feature_df,
        "feature_cols": feature_cols,
        "future_forecast_df": future_df,
        "train_df": train_df,
        "test_df": test_df,
        "cv_scores": cv_scores,
        "tuning_results": tuning_results,
        "ensemble_weights": ensemble_weights,
        "statistical_tests": stat_tests,
        "feature_importance": feature_importance,
        "confidence_intervals": conf_intervals
    }


def _build_future_feature_row(history_df: pd.DataFrame, next_date, feature_cols: list[str]) -> pd.DataFrame:
    hist = history_df.sort_values("Date").copy()
    values = hist["Demand"].astype(float).values

    def safe_lag(lag):
        return values[-lag] if len(values) >= lag else np.nan

    row = {
        "Lag1": safe_lag(1),
        "Lag7": safe_lag(7),
        "Lag14": safe_lag(14),
        "Lag30": safe_lag(30),
        "RollingMean7": np.mean(values[-7:]) if len(values) >= 7 else np.nan,
        "RollingMean30": np.mean(values[-30:]) if len(values) >= 30 else np.nan,
        "RollingStd7": np.std(values[-7:], ddof=1) if len(values) >= 7 else 0,
        "RollingStd30": np.std(values[-30:], ddof=1) if len(values) >= 30 else 0,
        "Year": next_date.year,
        "Month": next_date.month,
        "Quarter": (next_date.month - 1) // 3 + 1,
        "Week": int(next_date.isocalendar().week),
        "Weekday": next_date.weekday(),
        "IsWeekend": int(next_date.weekday() in [5, 6]),
        "DayOfYear": next_date.timetuple().tm_yday,
        "Trend": len(hist),
    }
    row["MonthSin"] = np.sin(2 * np.pi * row["Month"] / 12)
    row["MonthCos"] = np.cos(2 * np.pi * row["Month"] / 12)
    row["WeekSin"] = np.sin(2 * np.pi * row["Week"] / 52)
    row["WeekCos"] = np.cos(2 * np.pi * row["Week"] / 52)
    row["DayOfYearSin"] = np.sin(2 * np.pi * row["DayOfYear"] / 365.25)
    row["DayOfYearCos"] = np.cos(2 * np.pi * row["DayOfYear"] / 365.25)
    
    # EWMA simulation
    row["EWMA7"] = np.mean(values[-7:]) if len(values) >= 7 else np.nan
    row["EWMA30"] = np.mean(values[-30:]) if len(values) >= 30 else np.nan
    row["Diff1"] = (values[-1] - values[-2]) if len(values) >= 2 else 0
    row["Diff7"] = (values[-1] - values[-8]) if len(values) >= 8 else 0
    row["CV7"] = row["RollingStd7"] / (row["RollingMean7"] + 1e-9) if not np.isnan(row["RollingMean7"]) else 0
    row["Weekend_Lag1"] = row["IsWeekend"] * row["Lag1"]
    
    future_X = pd.DataFrame([row])
    for col in feature_cols:
        if col not in future_X.columns:
            future_X[col] = 0
    return future_X[feature_cols].ffill().fillna(0)


def _new_ml_model(model_name: str):
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=200, max_depth=16, min_samples_leaf=2, random_state=settings.random_state, n_jobs=-1
        )
    if model_name == "XGBoost" and XGBOOST_AVAILABLE:
        return XGBRegressor(n_estimators=300, learning_rate=0.04, max_depth=5, random_state=settings.random_state, n_jobs=-1)
    if model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
        return LGBMRegressor(n_estimators=400, learning_rate=0.03, random_state=settings.random_state, verbose=-1)
    return None


def _seasonal_naive_future(history_df: pd.DataFrame, horizon: int, season: int = 7) -> np.ndarray:
    values = history_df["Demand"].astype(float).values
    future = []
    for i in range(horizon):
        pred = values[-season + (i % season)] if len(values) >= season else np.mean(values)
        future.append(max(pred, 0))
    return np.array(future)


def _forecast_future(
    best_model_name: str,
    artifact: dict[str, Any],
    history_df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
) -> pd.DataFrame:
    history = history_df[["Date", "Demand"]].copy().sort_values("Date").reset_index(drop=True)
    full_features = create_time_series_features(history).dropna().reset_index(drop=True)
    last_date = history["Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    artifact_type = artifact.get("type")

    try:
        if artifact_type == "machine_learning":
            model = _new_ml_model(best_model_name)
            if model is None:
                raise ValueError("No ML model constructor")
            if artifact.get("model"):
                model.set_params(**artifact["model"].get_params())
                
            model.fit(full_features[feature_cols], full_features["Demand"])
            working_history = history.copy()
            preds = []
            for next_date in future_dates:
                X_next = _build_future_feature_row(working_history, next_date, feature_cols)
                pred = float(np.clip(model.predict(X_next)[0], 0, None))
                preds.append(pred)
                working_history = pd.concat(
                    [working_history, pd.DataFrame({"Date": [next_date], "Demand": [pred]})], ignore_index=True
                )
        elif artifact_type == "deep_learning" and HAS_TORCH:
            config = artifact["metadata"]["config"]
            scaler = artifact["metadata"]["scaler"]
            model_class = LSTMForecaster if best_model_name == "LSTM" else GRUForecaster
            
            preds = forecast_future_deep(
                artifact["model"], model_class, scaler, full_features[feature_cols], horizon, config
            )
            if preds is None:
                preds = _seasonal_naive_future(history, horizon)
                
        elif artifact_type == "statistical" and STATSMODELS_AVAILABLE:
            if best_model_name == "SARIMA":
                model = SARIMAX(history["Demand"], order=(1, 1, 1), seasonal_order=(1, 0, 1, 7))
            else:
                model = SARIMAX(history["Demand"], order=(2, 1, 2))
            fitted = model.fit(disp=False, maxiter=100)
            preds = np.clip(fitted.forecast(steps=horizon).values, 0, None)
        else:
            preds = _seasonal_naive_future(history, horizon)
    except Exception as e:
        logger.warning(f"Future forecast failed: {e}")
        preds = _seasonal_naive_future(history, horizon)

    return pd.DataFrame({"Date": future_dates, "ForecastDemand": preds})
