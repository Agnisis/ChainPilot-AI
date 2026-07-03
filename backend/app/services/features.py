import numpy as np
import pandas as pd

from app.services.data_loader import coerce_numeric


def create_time_series_features(df: pd.DataFrame, date_col: str = "Date", target_col: str = "Demand") -> pd.DataFrame:
    feature_df = df[[date_col, target_col]].copy()
    feature_df[date_col] = pd.to_datetime(feature_df[date_col], errors="coerce")
    feature_df[target_col] = coerce_numeric(feature_df[target_col])
    feature_df = feature_df.dropna(subset=[date_col, target_col]).sort_values(date_col).reset_index(drop=True)

    # 1. Standard Lags
    for lag in [1, 7, 14, 30]:
        feature_df[f"Lag{lag}"] = feature_df[target_col].shift(lag)

    # 2. Rolling Statistics (shifted by 1 to prevent data leakage)
    feature_df["RollingMean7"] = feature_df[target_col].shift(1).rolling(window=7).mean()
    feature_df["RollingMean30"] = feature_df[target_col].shift(1).rolling(window=30).mean()
    feature_df["RollingStd7"] = feature_df[target_col].shift(1).rolling(window=7).std()
    feature_df["RollingStd30"] = feature_df[target_col].shift(1).rolling(window=30).std()

    # 3. EWMA (Exponentially Weighted Moving Average)
    feature_df["EWMA7"] = feature_df[target_col].shift(1).ewm(span=7, adjust=False).mean()
    feature_df["EWMA30"] = feature_df[target_col].shift(1).ewm(span=30, adjust=False).mean()

    # 4. Differencing & Rate of Change
    feature_df["Diff1"] = feature_df[target_col].shift(1) - feature_df[target_col].shift(2)
    feature_df["Diff7"] = feature_df[target_col].shift(1) - feature_df[target_col].shift(8)
    
    # 5. Volatility / Coefficient of Variation
    feature_df["CV7"] = feature_df["RollingStd7"] / (feature_df["RollingMean7"] + 1e-9)

    # 6. Calendar Features
    dt = feature_df[date_col].dt
    iso = dt.isocalendar()
    feature_df["Year"] = dt.year
    feature_df["Month"] = dt.month
    feature_df["Quarter"] = dt.quarter
    feature_df["Week"] = iso.week.astype(int)
    feature_df["Weekday"] = dt.weekday
    feature_df["IsWeekend"] = feature_df["Weekday"].isin([5, 6]).astype(int)
    feature_df["DayOfYear"] = dt.dayofyear
    feature_df["Trend"] = np.arange(len(feature_df))

    # 7. Cyclical Encodings (Fourier-style)
    feature_df["MonthSin"] = np.sin(2 * np.pi * feature_df["Month"] / 12)
    feature_df["MonthCos"] = np.cos(2 * np.pi * feature_df["Month"] / 12)
    feature_df["WeekSin"] = np.sin(2 * np.pi * feature_df["Week"] / 52)
    feature_df["WeekCos"] = np.cos(2 * np.pi * feature_df["Week"] / 52)
    feature_df["DayOfYearSin"] = np.sin(2 * np.pi * feature_df["DayOfYear"] / 365.25)
    feature_df["DayOfYearCos"] = np.cos(2 * np.pi * feature_df["DayOfYear"] / 365.25)
    
    # 8. Interaction Features
    feature_df["Weekend_Lag1"] = feature_df["IsWeekend"] * feature_df["Lag1"]
    
    return feature_df
