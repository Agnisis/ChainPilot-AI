from typing import Any
from pydantic import BaseModel, Field

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    file_type: str
    message: str

class AnalysisRequest(BaseModel):
    session_id: str
    date_column: str | None = None
    target_column: str | None = None

class MetricRow(BaseModel):
    model: str
    mae: float
    rmse: float
    mape: float
    r2: float
    smape: float | None = None
    dir_acc: float | None = None
    rank: int
    is_ensemble: bool = False

class CVFoldResult(BaseModel):
    fold: int
    train_size: int
    test_size: int
    metrics: dict[str, float]

class TuningResult(BaseModel):
    model: str
    best_params: dict[str, Any]
    best_score: float
    n_trials: int

class SHAPFeature(BaseModel):
    feature: str
    importance: float
    direction: str | None = None

class EnsembleWeight(BaseModel):
    model: str
    weight: float

class ConfidenceInterval(BaseModel):
    date: str
    forecast: float
    lower_90: float
    upper_90: float
    lower_95: float
    upper_95: float

class StatisticalTest(BaseModel):
    test_name: str
    statistic: float
    p_value: float
    result: str
    interpretation: str

class ChartSeries(BaseModel):
    label: str
    data: list[float | int | None]
    borderColor: str | None = None
    backgroundColor: str | list[str] | None = None
    fill: bool | str | None = None

class ChartData(BaseModel):
    id: str
    title: str
    type: str  # line, bar, doughnut, radar, scatter
    labels: list[str]
    datasets: list[ChartSeries]
    interpretation: str = ""

class KPIItem(BaseModel):
    name: str
    value: str
    note: str = ""

class AnomalyDetail(BaseModel):
    date: str
    demand: float
    anomaly_type: str
    score: float
    severity: str
    detectors: list[str]
    root_cause_hint: str | None = None

class AnomalyPoint(BaseModel):
    date: str
    demand: float
    anomaly_type: str

class ForecastPoint(BaseModel):
    date: str
    forecast_demand: float

class AnalysisResult(BaseModel):
    session_id: str
    status: str
    forecasting_source: str
    best_model: str
    best_metrics: dict[str, float]
    model_ranking: list[MetricRow]
    cv_scores: dict[str, list[CVFoldResult]] | None = None
    tuning_results: list[TuningResult] | None = None
    ensemble_weights: list[EnsembleWeight] | None = None
    statistical_tests: list[StatisticalTest] | None = None
    feature_importance: list[SHAPFeature] | None = None
    confidence_intervals: list[ConfidenceInterval] | None = None
    forecast_summary: dict[str, float]
    kpis: list[KPIItem]
    charts: list[ChartData]
    anomalies: list[AnomalyPoint]
    anomaly_details: list[AnomalyDetail] | None = None
    anomaly_count: int
    message: str = ""

class RecommendationRequest(BaseModel):
    session_id: str
    question: str | None = None

class RecommendationResponse(BaseModel):
    session_id: str
    executive_summary: str
    key_insights: list[str]
    risks: list[str]
    inventory_recommendations: list[str]
    procurement_recommendations: list[str]
    logistics_recommendations: list[str]
    cost_optimization: list[str]
    strategic_path: str
    severity_scores: dict[str, str] | None = None
    rag_sources: list[str] | None = None
    raw_text: str

class RagQueryRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 5

class RagQueryResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[str]

class SessionStatus(BaseModel):
    session_id: str
    has_data: bool
    has_rag: bool
    analysis_ready: bool
    data_files: list[str]
    rag_files: list[str]
