from __future__ import annotations
"""
Hyperparameter tuning using Optuna Bayesian optimization.
Supports Random Forest, XGBoost, LightGBM tuning.
"""
import logging
import numpy as np
import pandas as pd
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("Optuna not installed — hyperparameter tuning unavailable")

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from app.config import settings

def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, n_trials: int = 30) -> dict:
    if not HAS_OPTUNA:
        return None
        
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 4, 32),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8, 1.0]),
            'random_state': settings.random_state,
            'n_jobs': -1
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=settings.random_state))
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Random Forest tuned: best RMSE={study.best_value:.4f}")
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'n_trials': n_trials
    }

def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, n_trials: int = 30) -> dict | None:
    if not HAS_OPTUNA:
        return None
        
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': settings.random_state,
            'n_jobs': -1
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=settings.random_state))
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"XGBoost tuned: best RMSE={study.best_value:.4f}")
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'n_trials': n_trials
    }

def tune_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, n_trials: int = 30) -> dict | None:
    if not HAS_OPTUNA:
        return None
        
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return None

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': settings.random_state,
            'verbose': -1
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=settings.random_state))
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"LightGBM tuned: best RMSE={study.best_value:.4f}")
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'n_trials': n_trials
    }

def tune_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, n_trials: int = 30) -> dict | None:
    if not HAS_OPTUNA or not settings.tuning_mode:
        return None
        
    try:
        if model_name == "Random Forest":
            return tune_random_forest(X_train, y_train, X_val, y_val, n_trials)
        elif model_name == "XGBoost":
            return tune_xgboost(X_train, y_train, X_val, y_val, n_trials)
        elif model_name == "LightGBM":
            return tune_lightgbm(X_train, y_train, X_val, y_val, n_trials)
    except Exception as e:
        logger.error(f"Failed to tune {model_name}: {e}")
        
    return None
