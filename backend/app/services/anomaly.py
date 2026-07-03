import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from app.config import settings
from app.services.data_loader import coerce_numeric, find_column


def detect_demand_anomalies(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Ensemble-based anomaly detection for demand spikes and drops."""
    anomaly_cols = [
        "Demand", "Lag1", "Lag7", "Lag14", "Lag30",
        "RollingMean7", "RollingMean30", "RollingStd7", "RollingStd30",
        "Diff1", "Diff7", "CV7"
    ]
    available_cols = [c for c in anomaly_cols if c in feature_df.columns]
    
    if len(available_cols) < 2 or len(feature_df) < 50:
        # Fallback for small data
        anomaly_df = feature_df.copy()
        anomaly_df["IsAnomaly"] = False
        anomaly_df["AnomalyType"] = "None"
        anomaly_df["Score"] = 0.0
        return anomaly_df

    anomaly_df = feature_df.dropna(subset=available_cols).copy()
    X = StandardScaler().fit_transform(anomaly_df[available_cols])
    
    # 1. Isolation Forest
    iso = IsolationForest(contamination=settings.anomaly_contamination, random_state=settings.random_state)
    iso_flags = iso.fit_predict(X)
    iso_scores = -iso.score_samples(X) # Higher means more anomalous
    
    # 2. Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=settings.anomaly_contamination)
    lof_flags = lof.fit_predict(X)
    lof_scores = -lof.negative_outlier_factor_
    
    # 3. One-Class SVM
    ocsvm = OneClassSVM(nu=settings.anomaly_contamination, kernel="rbf", gamma="scale")
    ocsvm_flags = ocsvm.fit_predict(X)
    # distance to separating hyperplane (more negative = more anomalous)
    ocsvm_scores = -ocsvm.score_samples(X) 
    
    # Normalize scores to [0, 1] for ensemble
    def norm(s):
        s = np.clip(s, np.percentile(s, 1), np.percentile(s, 99))
        if s.max() == s.min(): return np.zeros_like(s)
        return (s - s.min()) / (s.max() - s.min())
        
    combo_score = (norm(iso_scores) + norm(lof_scores) + norm(ocsvm_scores)) / 3.0
    anomaly_df["Score"] = combo_score
    
    # A point is an anomaly if at least 2 models agree, OR if combo score is very high
    votes = (iso_flags == -1).astype(int) + (lof_flags == -1).astype(int) + (ocsvm_flags == -1).astype(int)
    
    anomaly_df["IsAnomaly"] = (votes >= 2) | (combo_score > np.percentile(combo_score, 100 * (1 - settings.anomaly_contamination)))
    
    # Determine severity
    def get_severity(score):
        if score > 0.8: return "Critical"
        if score > 0.6: return "High"
        return "Medium"
        
    anomaly_df["Severity"] = anomaly_df["Score"].apply(lambda s: get_severity(s) if s > 0 else "Normal")
    
    # Track which detectors flagged it
    detectors = []
    for i in range(len(anomaly_df)):
        d = []
        if iso_flags[i] == -1: d.append("IsoForest")
        if lof_flags[i] == -1: d.append("LOF")
        if ocsvm_flags[i] == -1: d.append("OCSVM")
        detectors.append(d)
    anomaly_df["Detectors"] = detectors
    
    # Classify type
    rolling_col = "RollingMean30" if "RollingMean30" in anomaly_df.columns else "Demand"
    anomaly_df["AnomalyType"] = np.where(
        anomaly_df["Demand"] >= anomaly_df.get(rolling_col, anomaly_df["Demand"]),
        "Demand Spike",
        "Demand Drop"
    )
    
    # Add root cause hint
    hints = []
    for i, row in anomaly_df.iterrows():
        if not row["IsAnomaly"]:
            hints.append("")
            continue
            
        hint = []
        if "CV7" in anomaly_df.columns and row.get("CV7", 0) > anomaly_df["CV7"].mean() * 1.5:
            hint.append("High volatility period")
        if "Diff1" in anomaly_df.columns and abs(row.get("Diff1", 0)) > anomaly_df["Diff1"].std() * 2:
            hint.append("Sudden day-over-day change")
            
        hints.append(" | ".join(hint) if hint else "Unknown anomaly pattern")
        
    anomaly_df["RootCauseHint"] = hints
    
    return anomaly_df


def detect_inventory_anomalies(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    candidate_cols = []
    for preferred, tokens in [
        (["Order Item Quantity", "Quantity"], ["quantity"]),
        (["Sales"], ["sales"]),
        (["Benefit per order", "Order Profit Per Order"], ["profit", "benefit"]),
        (["Days for shipping (real)"], ["shipping"]),
    ]:
        col = find_column(df, preferred=preferred, contains_any=tokens)
        if col and col not in candidate_cols:
            candidate_cols.append(col)
    if len(candidate_cols) < 2:
        return pd.DataFrame()

    temp = df[candidate_cols].copy()
    for col in candidate_cols:
        temp[col] = coerce_numeric(temp[col])
    temp = temp.dropna()
    if len(temp) < 50:
        return pd.DataFrame()

    scaler = StandardScaler()
    X = scaler.fit_transform(temp)
    detector = IsolationForest(contamination=settings.anomaly_contamination, random_state=settings.random_state)
    flags = detector.fit_predict(X)
    result = df.loc[temp.index].copy()
    result["InventoryAnomaly"] = flags == -1
    return result[result["InventoryAnomaly"]].copy()
