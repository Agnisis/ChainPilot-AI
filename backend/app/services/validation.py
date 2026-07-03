from __future__ import annotations
"""
Time series validation: walk-forward CV, prediction intervals, statistical tests.
"""
import logging
import math
import numpy as np
import pandas as pd
from typing import Any, Callable
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)
from app.config import settings

def compute_extended_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes MAE, RMSE, MAPE, R2, SMAPE, and Directional Accuracy."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    min_len = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:min_len], y_pred[:min_len]
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    r2 = float(r2_score(y_true, y_pred))
    
    # SMAPE
    smape_denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape_denom = np.where(smape_denom < 1e-9, np.nan, smape_denom)
    smape = float(np.nanmean(np.abs(y_true - y_pred) / smape_denom) * 100)
    
    # Directional Accuracy (percentage of times direction matches)
    if len(y_true) > 1:
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        dir_match = (np.sign(true_diff) == np.sign(pred_diff)).mean() * 100
        dir_acc = float(dir_match)
    else:
        dir_acc = 100.0
        
    return {
        "MAE": mae, 
        "RMSE": rmse, 
        "MAPE": mape, 
        "R2": r2,
        "SMAPE": smape,
        "Directional_Accuracy": dir_acc
    }

def walk_forward_validation(
    model_factory: Callable, 
    df: pd.DataFrame, 
    feature_cols: list[str], 
    target_col: str = "Demand",
    n_splits: int = 5, 
    min_train_size: int = 60
) -> dict | None:
    """Expanding window time series cross-validation."""
    if len(df) < min_train_size + n_splits * 7:
        logger.warning(f"Dataset too small ({len(df)}) for walk-forward CV. Needs {min_train_size + n_splits * 7}.")
        return None
        
    X = df[feature_cols]
    y = df[target_col]
    
    fold_size = (len(df) - min_train_size) // n_splits
    if fold_size < 1:
        return None
        
    fold_results = []
    
    for fold in range(n_splits):
        train_end = min_train_size + fold * fold_size
        test_end = train_end + fold_size
        
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
        
        try:
            model = model_factory()
            if model is None:
                continue
                
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            preds = np.clip(preds, 0, None)
            
            metrics = compute_extended_metrics(y_test.values, preds)
            fold_results.append({
                "fold": fold + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "metrics": metrics
            })
        except Exception as e:
            logger.warning(f"Walk-forward CV fold {fold} failed: {e}")
            continue
            
    if not fold_results:
        return None
        
    # Aggregate metrics
    mean_metrics = {}
    for metric_name in fold_results[0]["metrics"].keys():
        vals = [f["metrics"][metric_name] for f in fold_results]
        mean_metrics[metric_name] = float(np.mean(vals))
        
    return {
        "fold_results": fold_results,
        "mean_metrics": mean_metrics
    }

def compute_prediction_intervals(
    y_pred: np.ndarray, 
    y_train_residuals: np.ndarray, 
    confidence_levels: list[float] = [0.90, 0.95]
) -> dict:
    """Bootstrap-based prediction intervals using historical residuals."""
    if len(y_train_residuals) < 10:
        return {}
        
    intervals = {}
    for conf in confidence_levels:
        alpha = 1.0 - conf
        lower_q = np.quantile(y_train_residuals, alpha / 2)
        upper_q = np.quantile(y_train_residuals, 1 - (alpha / 2))
        
        lower_bound = np.clip(y_pred + lower_q, 0, None)
        upper_bound = np.clip(y_pred + upper_q, 0, None)
        
        intervals[f"{int(conf*100)}"] = {
            "lower": lower_bound.tolist(),
            "upper": upper_bound.tolist()
        }
        
    return intervals

def adf_stationarity_test(series: pd.Series) -> dict | None:
    """Augmented Dickey-Fuller test for stationarity."""
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return None
        
    series = series.dropna()
    if len(series) < 20:
        return None
        
    try:
        result = adfuller(series)
        stat, p_value = result[0], result[1]
        is_stationary = p_value < 0.05
        
        return {
            "test_name": "Augmented Dickey-Fuller",
            "statistic": float(stat),
            "p_value": float(p_value),
            "result": "Stationary" if is_stationary else "Non-Stationary",
            "interpretation": "Time series has a stable mean/variance over time." if is_stationary else "Time series has trends/seasonality and needs differencing."
        }
    except Exception:
        return None

def ljung_box_residual_test(residuals: np.ndarray, lags: int = 10) -> dict | None:
    """Ljung-Box test for residual autocorrelation (white noise check)."""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        return None
        
    res = np.asarray(residuals)
    res = res[~np.isnan(res)]
    if len(res) < lags + 5:
        return None
        
    try:
        lb_results = acorr_ljungbox(res, lags=[lags], return_df=True)
        stat = lb_results.iloc[0]['lb_stat']
        p_value = lb_results.iloc[0]['lb_pvalue']
        is_white_noise = p_value > 0.05
        
        return {
            "test_name": "Ljung-Box (Residuals)",
            "statistic": float(stat),
            "p_value": float(p_value),
            "result": "White Noise" if is_white_noise else "Autocorrelated",
            "interpretation": "Model captured all patterns; residuals are random." if is_white_noise else "Model left signal in residuals; room for improvement."
        }
    except Exception:
        return None
