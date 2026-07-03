import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import settings
from app.models.schemas import AnalysisResult, AnomalyDetail, AnomalyPoint, ChartData, KPIItem, MetricRow
from app.services.anomaly import detect_demand_anomalies, detect_inventory_anomalies
from app.services.charts import build_charts
from app.services.data_loader import build_forecasting_frame, load_session_datasets
from app.services.features import create_time_series_features
from app.services.forecasting import run_forecasting_pipeline
from app.services.kpi import compute_supply_chain_kpis, format_kpi_value


class AnalysisPipeline:
    def run(
        self,
        session_id: str,
        date_column: str | None = None,
        target_column: str | None = None,
    ) -> AnalysisResult:
        session_data_dir = settings.data_dir / session_id
        if not session_data_dir.exists():
            raise FileNotFoundError(f"No data uploaded for session {session_id}")

        datasets = load_session_datasets(session_data_dir)
        forecasting_df, forecasting_source, dataco_df = build_forecasting_frame(
            datasets, date_column, target_column
        )

        forecast_output = run_forecasting_pipeline(forecasting_df)
        feature_df = forecast_output["feature_df"]
        demand_anomaly_df = detect_demand_anomalies(feature_df)
        demand_anomalies = demand_anomaly_df[demand_anomaly_df["IsAnomaly"]].copy()
        inventory_anomalies = detect_inventory_anomalies(dataco_df)
        kpis_raw = compute_supply_chain_kpis(dataco_df)

        charts = build_charts(
            forecasting_df=forecasting_df,
            feature_df=feature_df,
            metrics_df=forecast_output["metrics_df"],
            prediction_frames=forecast_output["prediction_frames"],
            future_forecast_df=forecast_output["future_forecast_df"],
            demand_anomalies=demand_anomalies,
            dataco_df=dataco_df,
            kpis=kpis_raw,
            best_model_name=forecast_output["best_model_name"],
        )

        forecast_summary = {
            f"{h}_day": float(forecast_output["future_forecast_df"].head(h)["ForecastDemand"].sum())
            for h in settings.forecast_horizons
        }

        kpi_items = [
            KPIItem(name="30-Day Forecast", value=f"{forecast_summary.get('30_day', 0):,.0f}", note="Near-term demand"),
            KPIItem(name="Best Model", value=forecast_output["best_model_name"], note="Lowest RMSE"),
            KPIItem(
                name="RMSE",
                value=f"{forecast_output['best_metrics']['RMSE']:,.2f}",
                note="Forecast accuracy",
            ),
            KPIItem(name="Anomalies", value=f"{len(demand_anomalies):,}", note="Demand exceptions"),
            KPIItem(
                name="Late Delivery Rate",
                value=format_kpi_value(kpis_raw.get("Late Delivery Rate"), pct=True),
                note="Shipment risk",
            ),
            KPIItem(
                name="Profit Margin",
                value=format_kpi_value(kpis_raw.get("Profit Margin"), pct=True),
                note="Profitability",
            ),
        ]

        metrics_rows = [
            MetricRow(
                model=row["Model"],
                mae=float(row["MAE"]),
                rmse=float(row["RMSE"]),
                mape=float(row["MAPE"]),
                r2=float(row["R2"]),
                smape=float(row.get("SMAPE", 0.0)),
                dir_acc=float(row.get("Directional_Accuracy", 0.0)),
                rank=int(row["Rank"]),
                is_ensemble="Ensemble" in row["Model"]
            )
            for _, row in forecast_output["metrics_df"].iterrows()
        ]

        anomaly_points = [
            AnomalyPoint(
                date=pd.Timestamp(r["Date"]).strftime("%Y-%m-%d"),
                demand=float(r["Demand"]),
                anomaly_type=str(r["AnomalyType"]),
            )
            for _, r in demand_anomalies.head(50).iterrows()
        ]
        
        # Format detailed anomalies if we have multi-detector output
        anomaly_details = None
        if "Score" in demand_anomalies.columns:
            anomaly_details = [
                AnomalyDetail(
                    date=pd.Timestamp(r["Date"]).strftime("%Y-%m-%d"),
                    demand=float(r["Demand"]),
                    anomaly_type=str(r["AnomalyType"]),
                    score=float(r["Score"]),
                    severity=str(r["Severity"]),
                    detectors=r["Detectors"] if isinstance(r["Detectors"], list) else [str(r["Detectors"])],
                    root_cause_hint=str(r.get("RootCauseHint", ""))
                )
                for _, r in demand_anomalies.head(50).iterrows()
            ]

        result = AnalysisResult(
            session_id=session_id,
            status="completed",
            forecasting_source=forecasting_source,
            best_model=forecast_output["best_model_name"],
            best_metrics=forecast_output["best_metrics"],
            model_ranking=metrics_rows,
            cv_scores=forecast_output.get("cv_scores"),
            tuning_results=forecast_output.get("tuning_results"),
            ensemble_weights=forecast_output.get("ensemble_weights"),
            statistical_tests=forecast_output.get("statistical_tests"),
            feature_importance=forecast_output.get("feature_importance"),
            confidence_intervals=forecast_output.get("confidence_intervals"),
            forecast_summary=forecast_summary,
            kpis=kpi_items,
            charts=charts,
            anomalies=anomaly_points,
            anomaly_details=anomaly_details,
            anomaly_count=len(demand_anomalies),
            message="Analysis completed successfully.",
        )

        self._save_result(session_id, result, kpis_raw, demand_anomalies, inventory_anomalies)
        return result

    def _save_result(
        self,
        session_id: str,
        result: AnalysisResult,
        kpis_raw: dict,
        demand_anomalies: pd.DataFrame,
        inventory_anomalies: pd.DataFrame,
    ) -> None:
        out_dir = settings.results_dir / session_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "analysis.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        summary = {
            "session_id": session_id,
            "forecasting_source": result.forecasting_source,
            "best_model": result.best_model,
            "best_metrics": result.best_metrics,
            "forecast_summary": result.forecast_summary,
            "anomaly_summary": {
                "demand_anomaly_count": result.anomaly_count,
                "inventory_anomaly_count": len(inventory_anomalies),
            },
            "supply_chain_kpis": kpis_raw,
            "model_ranking": [m.model_dump() for m in result.model_ranking],
            "feature_importance": [f.model_dump() for f in result.feature_importance[:5]] if result.feature_importance else None,
            "statistical_tests": [s.model_dump() for s in result.statistical_tests] if result.statistical_tests else None
        }
        (out_dir / "llm_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    def get_saved_result(self, session_id: str) -> AnalysisResult | None:
        path = settings.results_dir / session_id / "analysis.json"
        if not path.exists():
            return None
        return AnalysisResult.model_validate_json(path.read_text(encoding="utf-8"))

    def get_llm_summary(self, session_id: str) -> dict[str, Any] | None:
        path = settings.results_dir / session_id / "llm_summary.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


pipeline = AnalysisPipeline()
