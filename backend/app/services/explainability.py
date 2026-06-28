from __future__ import annotations
"""
Model explainability: SHAP values and feature importance.
"""
import logging
import numpy as np
import pandas as pd
from typing import Any

logger = logging.getLogger(__name__)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not installed — model explainability unavailable")

def compute_shap_values(model: Any, X_test: pd.DataFrame, model_type: str = "machine_learning") -> dict | None:
    """Computes SHAP feature importance for tree-based ML models."""
    if not HAS_SHAP or model_type != "machine_learning":
        return None
        
    try:
        # Check if it's a tree model
        if hasattr(model, 'estimators_') or hasattr(model, 'n_estimators') or model.__class__.__name__ in ['RandomForestRegressor', 'XGBRegressor', 'LGBMRegressor']:
            explainer = shap.TreeExplainer(model)
            # Use a background sample if X_test is too large to keep it fast
            X_sample = X_test.sample(min(100, len(X_test)), random_state=42) if len(X_test) > 100 else X_test
            shap_vals = explainer.shap_values(X_sample)
            
            # Aggregate importance
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            
            # Determine direction of impact
            # Correlation between feature value and SHAP value
            directions = []
            for i in range(X_sample.shape[1]):
                feat_vals = X_sample.iloc[:, i].values
                shap_col = shap_vals[:, i]
                # Avoid div by zero
                if np.std(feat_vals) > 0 and np.std(shap_col) > 0:
                    corr = np.corrcoef(feat_vals, shap_col)[0, 1]
                    if corr > 0.3:
                        dir_str = "Positive (Higher value -> Higher demand)"
                    elif corr < -0.3:
                        dir_str = "Negative (Higher value -> Lower demand)"
                    else:
                        dir_str = "Complex / Non-linear"
                else:
                    dir_str = "Neutral"
                directions.append(dir_str)
                
            features = []
            for i, feat in enumerate(X_test.columns):
                features.append({
                    "feature": feat,
                    "importance": float(mean_abs_shap[i]),
                    "direction": directions[i]
                })
                
            # Sort by importance
            features.sort(key=lambda x: x["importance"], reverse=True)
            
            return {
                "features": features,
                "base_value": float(explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
            }
            
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        
    return None

def generate_model_explanation_text(shap_results: dict | None, model_name: str) -> str:
    """Generates a natural language explanation of feature importance for the LLM context."""
    if not shap_results or not shap_results.get("features"):
        return f"No interpretability data available for {model_name}."
        
    top_features = shap_results["features"][:5]
    
    explanation = f"The {model_name} forecasting model relies most heavily on these top factors:\n"
    for i, f in enumerate(top_features):
        explanation += f"{i+1}. {f['feature']} (Impact score: {f['importance']:.2f}) - {f['direction']}\n"
        
    return explanation
