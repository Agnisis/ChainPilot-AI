from __future__ import annotations
"""
Ensemble methods: stacking, weighted averaging for model combination.
"""
import logging
import numpy as np
from typing import Any
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

def build_stacking_ensemble(
    base_predictions: dict[str, np.ndarray], 
    y_true: np.ndarray, 
    test_predictions: dict[str, np.ndarray] | None = None
) -> dict:
    """Trains a Ridge meta-model on base predictions to find optimal combination weights."""
    models = list(base_predictions.keys())
    if len(models) < 2:
        return {"predictions": list(base_predictions.values())[0], "weights": {models[0]: 1.0}}
        
    X_stack_train = np.column_stack([base_predictions[m] for m in models])
    
    meta_model = Ridge(alpha=1.0, positive=True) # positive=True ensures no negative weights
    meta_model.fit(X_stack_train, y_true)
    
    # Normalize weights to sum to 1
    weights = meta_model.coef_
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        weights = np.ones(len(models)) / len(models)
        
    weight_dict = {m: float(w) for m, w in zip(models, weights)}
    
    result = {
        "meta_model": meta_model,
        "weights": weight_dict,
        "models_used": models
    }
    
    if test_predictions:
        X_stack_test = np.column_stack([test_predictions[m] for m in models])
        preds = meta_model.predict(X_stack_test)
        result["predictions"] = np.clip(preds, 0, None)
        
    return result

def weighted_average_ensemble(
    predictions: dict[str, np.ndarray], 
    metrics: dict[str, float], 
    metric_key: str = "RMSE"
) -> dict:
    """Combines predictions weighted inversely by their RMSE/MAE error."""
    models = list(predictions.keys())
    if len(models) < 2:
        return {"predictions": list(predictions.values())[0], "weights": {models[0]: 1.0}}
        
    # Calculate inverse error weights
    errors = np.array([metrics[m] for m in models])
    # Add small epsilon to avoid divide by zero
    inv_errors = 1.0 / (errors + 1e-9)
    weights = inv_errors / np.sum(inv_errors)
    
    weight_dict = {m: float(w) for m, w in zip(models, weights)}
    
    # Generate weighted predictions
    ensemble_preds = np.zeros_like(predictions[models[0]], dtype=float)
    for i, m in enumerate(models):
        ensemble_preds += predictions[m] * weights[i]
        
    return {
        "predictions": np.clip(ensemble_preds, 0, None),
        "weights": weight_dict,
        "models_used": models
    }
