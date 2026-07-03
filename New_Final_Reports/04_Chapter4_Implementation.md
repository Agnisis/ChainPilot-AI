# CHAPTER 4

## IMPLEMENTATION DETAILS

This chapter provides the detailed mathematical formulations, algorithmic logic, and Python implementation specifics for each of the six core AI models deployed within ChainPilot AI. All implementations use the exact libraries, class names, and hyperparameters from the production codebase.

### 4.1 Data Preprocessing and Feature Engineering

Before any model can be trained, the raw time-series demand data must be transformed into a rich feature matrix. The `features.py` module implements the `create_time_series_features()` function, which engineers **28 predictive features** across 8 categories from the raw Date/Demand series.

**Table 4.1: Engineered Feature Set (28 Features)**

| Category | Features | Mathematical Definition |
|----------|----------|------------------------|
| **Lag Features** | `Lag1`, `Lag7`, `Lag14`, `Lag30` | $X_{t-k}$ for $k \in \{1, 7, 14, 30\}$ |
| **Rolling Statistics** | `RollingMean7`, `RollingMean30`, `RollingStd7`, `RollingStd30` | $\bar{X}_{t,w} = \frac{1}{w}\sum_{i=1}^{w}X_{t-i}$ (shifted by 1 to prevent leakage) |
| **Exponential Weighted** | `EWMA7`, `EWMA30` | $S_t = \alpha X_{t-1} + (1-\alpha)S_{t-1}$, where $\alpha = 2/(w+1)$ |
| **Differencing** | `Diff1`, `Diff7` | $\Delta X_t = X_t - X_{t-1}$ (day-over-day), $\Delta_7 X_t = X_t - X_{t-7}$ (week-over-week) |
| **Volatility** | `CV7` | Coefficient of Variation: $CV = \sigma_7 / \bar{X}_7$ |
| **Calendar** | `Year`, `Month`, `Quarter`, `Week`, `Weekday`, `IsWeekend`, `DayOfYear`, `Trend` | Extracted from the datetime index |
| **Cyclical (Fourier)** | `MonthSin`, `MonthCos`, `WeekSin`, `WeekCos`, `DayOfYearSin`, `DayOfYearCos` | $\sin(2\pi \cdot \text{period}/\text{max\_period})$, $\cos(2\pi \cdot \text{period}/\text{max\_period})$ |
| **Interaction** | `Weekend_Lag1` | $\text{IsWeekend} \times \text{Lag1}$ |

A critical design decision in the feature engineering pipeline is the **shift-by-one** applied to all rolling statistics and exponentially weighted moving averages. Without this shift, the rolling mean at time $t$ would include the value $X_t$ itself, creating a subtle but devastating data leakage that artificially inflates model accuracy during training.

### 4.2 Auto-ARIMA Implementation

The Auto-ARIMA model is implemented using the `pmdarima` library, which automates the Box-Jenkins methodology by systematically searching across combinations of $(p, d, q)$ and seasonal $(P, D, Q, m)$ parameters using the Akaike Information Criterion (AIC).

**Configuration:**
```python
model = pm.auto_arima(
    y_train,
    seasonal=True,
    m=7,                    # Weekly seasonality
    stepwise=True,          # Efficient stepwise search
    suppress_warnings=True,
    error_action='ignore',
    trace=False
)
```

The `m=7` parameter explicitly encodes weekly seasonality, which is the dominant periodic pattern in retail and logistics demand data. The stepwise search algorithm reduces computational cost compared to exhaustive grid search by evaluating only the most promising parameter combinations based on AIC improvements.

For the test set, predictions are generated iteratively using one-step-ahead forecasting with `model.predict(n_periods=len(y_test))`. The resulting predictions are then evaluated against the held-out test set using RMSE, MAE, MAPE, and Directional Accuracy.

### 4.3 Random Forest Regressor

The Random Forest implementation uses `scikit-learn`'s `RandomForestRegressor` with the following production configuration:

```python
model = RandomForestRegressor(
    n_estimators=300,       # 300 independent decision trees
    max_depth=16,           # Maximum tree depth
    min_samples_leaf=2,     # Minimum samples at leaf node
    random_state=42,        # Reproducibility
    n_jobs=-1               # Parallel training across all CPU cores
)
```

The model is trained on the 28-feature matrix described in Section 4.1. Unlike ARIMA, Random Forest treats each time step as an independent observation vector, relying entirely on the engineered lag and rolling features to capture temporal dependencies.

After training, **SHAP (SHapley Additive exPlanations)** values are computed using `shap.TreeExplainer` to determine the feature importance ranking. The SHAP analysis samples a maximum of 100 test observations for computational efficiency and computes the mean absolute SHAP value for each feature. The direction of influence (positive or negative) is determined by the Pearson correlation between the feature values and their corresponding SHAP values, using a threshold of $\pm 0.3$.

### 4.4 XGBoost with Optuna Bayesian Optimization

XGBoost is the most heavily optimized model in the pipeline. Instead of relying on default hyperparameters or manual tuning, the system employs Optuna's Tree-structured Parzen Estimator (TPE) algorithm for Bayesian hyperparameter optimization.

**Default Configuration (before Optuna tuning):**
```python
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.04,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
```

**Optuna Search Space:**

**Table 4.2: Optuna Hyperparameter Search Space for XGBoost**

| Hyperparameter | Type | Range | Scale |
|----------------|------|-------|-------|
| `n_estimators` | Integer | 100 - 600 | Step = 50 |
| `learning_rate` | Float | 0.01 - 0.15 | Logarithmic |
| `max_depth` | Integer | 3 - 10 | Linear |
| `subsample` | Float | 0.6 - 1.0 | Linear |
| `colsample_bytree` | Float | 0.5 - 1.0 | Linear |
| `reg_alpha` | Float | 1e-8 - 10.0 | Logarithmic |
| `reg_lambda` | Float | 1e-8 - 10.0 | Logarithmic |
| `min_child_weight` | Integer | 1 - 10 | Linear |

The optimization uses `TPESampler(seed=42)` for reproducibility and minimizes RMSE on the validation set across a configurable number of trials (default: 5 trials). The Bayesian nature of TPE means that each subsequent trial is informed by the results of all previous trials, concentrating the search in the most promising regions of the hyperparameter space.

### 4.5 PyTorch LSTM Forecaster

The Long Short-Term Memory (LSTM) implementation is the most architecturally complex component of the pipeline. It is implemented as a custom PyTorch `nn.Module` class:

```python
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,     # Number of features (28)
            hidden_size=hidden_size,   # Hidden state dimension (64)
            num_layers=num_layers,     # Stacked LSTM layers (2)
            batch_first=True,          # Input shape: (batch, seq, features)
            dropout=dropout            # Dropout between layers
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)  # Output: single demand value

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # Take last time step
        return self.fc(self.dropout(last_out))
```

**Table 4.3: PyTorch LSTM/GRU Training Configuration**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Sequence Length | 30 | Input window (30 days of history) |
| Hidden Size | 64 | Dimensionality of hidden state |
| Number of Layers | 2 | Stacked recurrent layers |
| Dropout Rate | 0.2 | Regularization between layers |
| Learning Rate | 0.001 | Adam optimizer initial LR |
| Batch Size | 32 | Mini-batch gradient descent |
| Epochs | 20 | Maximum training iterations |
| Early Stopping Patience | 10 | Epochs without improvement before stopping |
| LR Scheduler | ReduceLROnPlateau | Halves LR after 5 epochs of no improvement |
| Gradient Clipping | 1.0 | Prevents exploding gradients |
| Loss Function | MSELoss | Mean Squared Error |

**Data Preparation:**
The raw feature matrix is transformed into 3D tensors using a custom `TimeSeriesDataset` class that creates sliding windows of `sequence_length=30`. For each window, the input is a tensor of shape `(30, 28)` (30 time steps x 28 features), and the target is the demand value at time step 31. All features are standardized using `sklearn.preprocessing.StandardScaler` before being fed to the network.

**Training Loop:**
The training procedure implements several best practices for deep learning:
1. **Mini-batch gradient descent** with `DataLoader(batch_size=32, shuffle=True)`
2. **Adam optimizer** with adaptive learning rates
3. **ReduceLROnPlateau scheduler** that halves the learning rate after 5 epochs of no validation loss improvement
4. **Gradient clipping** (`clip_grad_norm_(max_norm=1.0)`) to prevent the exploding gradient problem
5. **Early stopping** that saves the best model state and restores it after `patience=10` epochs of no improvement

**Future Forecasting:**
For generating 90-day future predictions, the system uses an autoregressive approach: the model predicts day $t+1$, then constructs a new feature row for day $t+1$ (using the predicted value as the new lag), shifts the sliding window forward by one step, and repeats until the full 90-day horizon is reached.

### 4.6 PyTorch GRU Forecaster

The Gated Recurrent Unit (GRU) is implemented as a computationally lighter alternative to the LSTM, merging the forget and input gates into a single update gate:

```python
class GRUForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        return self.fc(self.dropout(last_out))
```

The GRU uses the identical training configuration, data preparation, and evaluation methodology as the LSTM (Table 4.3). The architectural difference is that the GRU has approximately 25% fewer parameters than the LSTM (due to having 2 gates instead of 3), resulting in faster training times with comparable prediction accuracy.

### 4.7 Isolation Forest Anomaly Detection

The anomaly detection module implements a sophisticated **three-detector ensemble** that goes far beyond a simple single-algorithm approach:

**Detector 1: Isolation Forest**
```python
IsolationForest(contamination=0.03, random_state=42)
```

**Detector 2: Local Outlier Factor (LOF)**
```python
LocalOutlierFactor(n_neighbors=20, contamination=0.03)
```

**Detector 3: One-Class SVM**
```python
OneClassSVM(nu=0.03, kernel="rbf", gamma="scale")
```

**Ensemble Scoring Logic:**
Each detector produces a raw anomaly score. These scores are individually normalized to the $[0, 1]$ range and then averaged into a composite anomaly score. A data point is flagged as anomalous if **at least 2 out of 3 detectors agree** OR if the composite score exceeds a dynamic threshold.

**Severity Classification:**
- **Critical** ($\text{score} > 0.8$): Immediate executive escalation
- **High** ($\text{score} > 0.6$): Requires investigation within 24 hours
- **Medium** (else): Standard monitoring

**Type Classification:** Anomalies are classified as "Spike" (demand significantly above `RollingMean30`) or "Drop" (demand significantly below `RollingMean30`), enabling differentiated response strategies.

### 4.8 Walk-Forward Cross-Validation

To rigorously evaluate model generalization without data leakage, the system implements **expanding-window walk-forward cross-validation**. Unlike standard k-fold cross-validation (which randomly shuffles data and violates temporal ordering), walk-forward CV respects the chronological structure of time-series data.

For each fold $k$ (default: 3 folds):
1. The training set expands from the beginning of the data up to fold boundary $k$
2. The test set is the next `fold_size` observations immediately after the training boundary
3. The model is retrained from scratch on the expanded training set
4. Predictions are generated on the unseen test set
5. Metrics (RMSE, MAE, MAPE, R², Directional Accuracy) are computed

The final reported metrics are the mean across all folds, providing a robust estimate of the model's expected performance on truly unseen future data.

**Evaluation Metrics:**

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

$$\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

$$\text{SMAPE} = \frac{200\%}{n}\sum_{i=1}^{n}\frac{|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}$$

$$\text{Directional Accuracy} = \frac{1}{n-1}\sum_{i=2}^{n}\mathbb{1}[\text{sign}(\Delta y_i) = \text{sign}(\Delta \hat{y}_i)]$$

All six metrics are computed for every model, enabling a comprehensive multi-dimensional comparison that goes beyond single-metric model selection.
