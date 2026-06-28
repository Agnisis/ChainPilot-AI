$ErrorActionPreference = "Stop"

$cells = New-Object System.Collections.Generic.List[object]

function Add-Markdown {
    param([Parameter(Mandatory=$true)][string]$Source)
    $cells.Add([ordered]@{
        cell_type = "markdown"
        metadata = [ordered]@{}
        source = $Source.TrimStart("`r", "`n")
    }) | Out-Null
}

function Add-Code {
    param([Parameter(Mandatory=$true)][string]$Source)
    $cells.Add([ordered]@{
        cell_type = "code"
        execution_count = $null
        metadata = [ordered]@{}
        outputs = @()
        source = $Source.TrimStart("`r", "`n")
    }) | Out-Null
}

Add-Markdown @'
# AI Powered Supply Chain Intelligence Platform

**Notebook prototype for demand forecasting, anomaly detection, supply chain intelligence, and LLM-assisted executive recommendations.**

This notebook is designed as a Master's thesis / academic review / industry demonstration artifact. It is intentionally modular so the same logic can later be refactored into a Python package, dashboard service, API backend, or SaaS platform.

**Primary datasets supported**

- M5 Forecasting dataset: `calendar.csv`, `sales_train_validation.csv`, `sell_prices.csv`
- DataCo Supply Chain dataset: `DataCoSupplyChainDataset.csv`

**Local path strategy**

The notebook searches both `.data/` and `Data/` recursively. In this workspace the datasets are currently under `Data/`, while the project requirement names `.data/`; supporting both keeps the notebook portable.
'@

Add-Markdown @'
## 1. Project Introduction

### Problem Statement

Modern supply chains often operate reactively. Demand is forecast using manual spreadsheets, supplier risk is discovered after failures occur, shipment delays are handled after customers are already impacted, and inventory decisions are frequently disconnected from real-time market signals.

This project builds an AI-powered supply chain intelligence prototype that moves decision making from reactive to predictive and prescriptive.

### Business Problem

Organizations face several recurring operational problems:

- Demand uncertainty causes stockouts, lost sales, and excess safety stock.
- Overstocking ties up working capital and increases warehousing costs.
- Shipment delays reduce service levels and customer trust.
- Supplier and logistics risks are hard to monitor continuously.
- Teams lack a unified intelligence layer that combines forecasting, anomaly detection, KPIs, and recommendations.

### Research Gap

Many academic and industry workflows treat forecasting, anomaly detection, KPI reporting, and executive recommendations as isolated tasks. Real supply chain decisions require these components to work together:

- Forecasting identifies expected future demand.
- Anomaly detection identifies abnormal deviations and operational shocks.
- Supply chain KPIs explain service, profit, shipment, and risk exposure.
- LLM-based reasoning converts technical outputs into business-facing recommendations.

This notebook demonstrates an integrated research prototype that connects these layers.

### Project Objectives

- Ingest and profile M5 and DataCo supply chain datasets.
- Engineer time-series, lag, rolling, trend, and seasonality features.
- Train statistical, machine learning, and deep learning forecasting models.
- Benchmark models using MAE, RMSE, MAPE, and R2.
- Select the best model automatically using lowest RMSE.
- Forecast future demand for 30, 60, and 90 days.
- Detect demand, inventory, and shipment anomalies.
- Generate supply chain KPIs and executive dashboard summaries.
- Use a free-tier compatible LLM workflow through Google Gemini for recommendations.

### Expected Benefits

- Better inventory planning and reduced stockout risk.
- Lower overstocking and carrying cost.
- Earlier detection of demand spikes, demand drops, and shipment disruption.
- Faster executive decision making through automated intelligence summaries.
- A clear prototype path toward a production SaaS platform.

### Future SaaS Vision

The prototype can evolve into a SaaS platform integrating OMS, WMS, TMS, ERP, inventory, supplier, and streaming systems through APIs. The future product can support multi-tenant dashboards, automated alerts, RAG over supply chain documents, and agentic AI workflows for procurement, logistics, and inventory planning.
'@

Add-Markdown @'
## 2. Library Installation

Run the next cell once in a fresh environment. It installs the analytics, forecasting, visualization, deep learning, and LLM packages required by the notebook.

If you already have these packages installed, you may skip the cell.
'@

Add-Code @'
# Install required packages.
# In Jupyter this uses the active kernel environment, which is safer than calling a system-level pip.
%pip install -q pandas numpy matplotlib plotly seaborn scikit-learn statsmodels xgboost lightgbm tensorflow keras prophet google-generativeai
'@

Add-Markdown @'
## 3. Import Libraries

Imports are grouped by purpose. Optional libraries are handled gracefully so the notebook remains executable even if a heavy dependency is unavailable in a specific environment.
'@

Add-Code @'
import os
import re
import json
import math
import warnings
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from IPython.display import display, Markdown, HTML

from sklearn.base import clone
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 120)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

sns.set_theme(style="whitegrid", context="notebook")
PLOTLY_TEMPLATE = "plotly_white"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Optional forecasting libraries.
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except Exception as exc:
    STATSMODELS_AVAILABLE = False
    print(f"statsmodels unavailable: {exc}")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception as exc:
    XGBOOST_AVAILABLE = False
    print(f"xgboost unavailable: {exc}")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except Exception as exc:
    LIGHTGBM_AVAILABLE = False
    print(f"lightgbm unavailable: {exc}")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
    tf.random.set_seed(RANDOM_STATE)
except Exception as exc:
    TENSORFLOW_AVAILABLE = False
    print(f"tensorflow unavailable: {exc}")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception as exc:
    GEMINI_AVAILABLE = False
    print(f"google-generativeai unavailable: {exc}")

display(Markdown("**Environment initialized.** Optional libraries are flagged above if unavailable."))
'@

Add-Markdown @'
## 4. Data Loading

This section loads local datasets, discovers paths automatically, displays schema snapshots, and prepares a daily demand time series for forecasting.

The notebook supports explicit configuration through `DATE_COLUMN` and `TARGET_COLUMN`. If those are left as `None`, the notebook detects suitable columns from the data.
'@

Add-Code @'
# -----------------------------
# Global configuration
# -----------------------------

PROJECT_NAME = "AI Powered Supply Chain Intelligence Platform"

# The user requirement names ".data"; this workspace currently uses "Data".
# The loader searches both recursively.
DATA_ROOT_CANDIDATES = [
    Path(".data"),
    Path("Data"),
    Path("data"),
    Path("."),
]

# Optional overrides for generic datasets. Leave as None for automatic detection.
DATE_COLUMN = None
TARGET_COLUMN = None

# Runtime controls.
FAST_MODE = True
M5_PRICE_SAMPLE_ROWS = 250_000 if FAST_MODE else None
DATACO_SAMPLE_ROWS = None

TEST_SIZE_DAYS = 90
FORECAST_HORIZONS = [30, 60, 90]
DEEP_LEARNING_EPOCHS = 8 if FAST_MODE else 30
DEEP_LEARNING_BATCH_SIZE = 32
SEQUENCE_LENGTH = 30
ANOMALY_CONTAMINATION = 0.03

print(f"Project: {PROJECT_NAME}")
print(f"Search roots: {[str(p) for p in DATA_ROOT_CANDIDATES]}")
'@

Add-Code @'
# -----------------------------
# Reusable data loading helpers
# -----------------------------

def normalize_name(name):
    """Normalize a column name for robust matching across messy source systems."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def find_file(filename, roots=DATA_ROOT_CANDIDATES):
    """Find the first matching file recursively under the configured data roots."""
    for root in roots:
        if root.exists():
            matches = sorted(root.rglob(filename))
            if matches:
                return matches[0]
    return None


def read_csv_resilient(path, nrows=None, low_memory=False):
    """Read CSV with encoding fallback. This is useful for Kaggle-style datasets."""
    if path is None:
        return None
    encodings = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, nrows=nrows, low_memory=low_memory)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def display_dataset_overview(df, name, sample_rows=5):
    """Display shape, columns, head, and random sample for review readiness."""
    if df is None:
        display(Markdown(f"### {name}\nDataset not found or not loaded."))
        return
    display(Markdown(f"### {name}"))
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    print("Columns:")
    display(pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]}))
    display(Markdown("**Head**"))
    display(df.head(sample_rows))
    display(Markdown("**Random sample**"))
    display(df.sample(min(sample_rows, len(df)), random_state=RANDOM_STATE))


def find_column(df, preferred=None, contains_any=None, contains_all=None):
    """Find a column by exact normalized names or token matching."""
    if df is None or df.empty:
        return None
    preferred = preferred or []
    contains_any = contains_any or []
    contains_all = contains_all or []
    normalized_to_original = {normalize_name(c): c for c in df.columns}
    for candidate in preferred:
        key = normalize_name(candidate)
        if key in normalized_to_original:
            return normalized_to_original[key]
    for col in df.columns:
        n = normalize_name(col)
        if contains_all and all(token in n for token in contains_all):
            return col
        if contains_any and any(token in n for token in contains_any):
            return col
    return None


def coerce_numeric(series):
    """Convert potentially formatted numeric columns to numeric values."""
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")
'@

Add-Code @'
# -----------------------------
# Locate and load source files
# -----------------------------

file_paths = {
    "calendar": find_file("calendar.csv"),
    "sales_train_validation": find_file("sales_train_validation.csv"),
    "sell_prices": find_file("sell_prices.csv"),
    "dataco": find_file("DataCoSupplyChainDataset.csv"),
}

display(pd.DataFrame(
    [{"dataset": key, "path": str(value) if value else "NOT FOUND"} for key, value in file_paths.items()]
))

calendar_df = read_csv_resilient(file_paths["calendar"], low_memory=False)
sales_df = read_csv_resilient(file_paths["sales_train_validation"], low_memory=False)
sell_prices_df = read_csv_resilient(file_paths["sell_prices"], nrows=M5_PRICE_SAMPLE_ROWS, low_memory=False)
dataco_df = read_csv_resilient(file_paths["dataco"], nrows=DATACO_SAMPLE_ROWS, low_memory=False)

display_dataset_overview(calendar_df, "M5 calendar.csv")
display_dataset_overview(sales_df, "M5 sales_train_validation.csv")
display_dataset_overview(sell_prices_df, "M5 sell_prices.csv")
display_dataset_overview(dataco_df, "DataCoSupplyChainDataset.csv")
'@

Add-Code @'
# -----------------------------
# Prepare forecasting-ready time series
# -----------------------------

def prepare_m5_timeseries(calendar, sales):
    """Aggregate M5 item-level sales into a daily total demand time series."""
    if calendar is None or sales is None:
        return None

    d_cols = sorted(
        [c for c in sales.columns if re.fullmatch(r"d_\d+", str(c))],
        key=lambda x: int(x.split("_")[1])
    )
    if not d_cols:
        return None

    demand_by_d = (
        sales[d_cols]
        .sum(axis=0)
        .rename("Demand")
        .reset_index()
        .rename(columns={"index": "d"})
    )

    calendar_cols = [c for c in ["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year", "event_name_1", "event_type_1"] if c in calendar.columns]
    calendar_small = calendar[calendar_cols].copy()
    calendar_small["date"] = pd.to_datetime(calendar_small["date"], errors="coerce")

    ts = calendar_small.merge(demand_by_d, on="d", how="inner")
    ts = ts.rename(columns={"date": "Date"})
    ts = ts.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return ts


def detect_date_target_columns(df, date_override=None, target_override=None):
    """Detect date and target columns for generic order/sales datasets."""
    if df is None:
        return None, None

    date_col = date_override or find_column(
        df,
        preferred=["date", "order date", "order date (DateOrders)", "shipping date (DateOrders)"],
        contains_any=["dateorders", "order date", "date"]
    )

    target_col = target_override or find_column(
        df,
        preferred=["Sales", "Order Item Quantity", "Order Item Total", "Demand", "Quantity"],
        contains_any=["sales", "quantity", "demand"]
    )
    return date_col, target_col


def prepare_generic_timeseries(df, date_col=None, target_col=None):
    """Aggregate a generic transactional dataset into daily demand."""
    if df is None:
        return None
    date_col, target_col = detect_date_target_columns(df, date_col, target_col)
    if date_col is None or target_col is None:
        return None

    temp = df[[date_col, target_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[target_col] = coerce_numeric(temp[target_col])
    temp = temp.dropna(subset=[date_col, target_col])

    daily = (
        temp
        .groupby(pd.Grouper(key=date_col, freq="D"))[target_col]
        .sum()
        .reset_index()
        .rename(columns={date_col: "Date", target_col: "Demand"})
    )
    daily = daily[daily["Demand"].notna()].sort_values("Date").reset_index(drop=True)
    return daily


m5_ts = prepare_m5_timeseries(calendar_df, sales_df)
dataco_ts = prepare_generic_timeseries(dataco_df, DATE_COLUMN, TARGET_COLUMN)

if m5_ts is not None and len(m5_ts) >= 180:
    forecasting_df = m5_ts[["Date", "Demand"]].copy()
    forecasting_source = "M5 aggregated daily demand"
elif dataco_ts is not None and len(dataco_ts) >= 180:
    forecasting_df = dataco_ts[["Date", "Demand"]].copy()
    forecasting_source = "DataCo aggregated daily demand"
else:
    raise ValueError("No forecasting-ready dataset found. Check dataset paths and DATE_COLUMN / TARGET_COLUMN settings.")

display(Markdown(f"**Forecasting source selected:** {forecasting_source}"))
display(forecasting_df.head())
display(forecasting_df.tail())
print(f"Forecasting rows: {len(forecasting_df):,}")
'@

Add-Markdown @'
## 5. Data Quality Analysis

Data quality checks are essential before forecasting because missing dates, duplicated transactions, outliers, and mixed data types can produce misleading demand signals.

This section covers:

- Missing values analysis
- Duplicate analysis
- Outlier analysis using IQR
- Data type analysis
- Visual summaries for review
'@

Add-Code @'
def data_quality_report(df, name):
    """Create a reusable quality report for any dataframe."""
    if df is None:
        return None

    missing = (
        df.isna().sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing["missing_pct"] = missing["missing_count"] / max(len(df), 1) * 100
    missing = missing.sort_values("missing_pct", ascending=False)

    duplicate_count = int(df.duplicated().sum())
    dtype_summary = (
        pd.Series([str(t) for t in df.dtypes])
        .value_counts()
        .rename_axis("dtype")
        .reset_index(name="column_count")
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_rows = []
    for col in numeric_cols[:50]:
        values = df[col].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((values < lower) | (values > upper)).sum())
        outlier_rows.append({
            "column": col,
            "outlier_count": count,
            "outlier_pct": count / max(len(values), 1) * 100,
            "lower_bound": lower,
            "upper_bound": upper,
        })
    outliers = pd.DataFrame(outlier_rows).sort_values("outlier_pct", ascending=False) if outlier_rows else pd.DataFrame()

    report = {
        "name": name,
        "shape": df.shape,
        "missing": missing,
        "duplicate_count": duplicate_count,
        "dtype_summary": dtype_summary,
        "outliers": outliers,
    }
    return report


def visualize_quality_report(report):
    """Visualize missing values, data types, and outliers."""
    if report is None:
        return
    name = report["name"]
    display(Markdown(f"### Data Quality Report: {name}"))
    print(f"Shape: {report['shape'][0]:,} rows x {report['shape'][1]:,} columns")
    print(f"Duplicate rows: {report['duplicate_count']:,}")

    missing_top = report["missing"].query("missing_count > 0").head(20)
    if not missing_top.empty:
        fig = px.bar(
            missing_top,
            x="missing_pct",
            y="column",
            orientation="h",
            title=f"{name}: Top Missing Value Percentages",
            labels={"missing_pct": "Missing (%)", "column": "Column"},
            template=PLOTLY_TEMPLATE,
        )
        fig.show()
        display(Markdown("**Business interpretation:** Columns with high missingness may reduce visibility and should be treated before operational decision making."))
    else:
        display(Markdown("**Missing values:** No missing values detected."))

    fig = px.bar(
        report["dtype_summary"],
        x="dtype",
        y="column_count",
        title=f"{name}: Data Type Distribution",
        template=PLOTLY_TEMPLATE,
    )
    fig.show()
    display(Markdown("**Business interpretation:** Mixed types often indicate source-system inconsistencies that must be standardized before production integration."))

    if not report["outliers"].empty:
        fig = px.bar(
            report["outliers"].head(20),
            x="outlier_pct",
            y="column",
            orientation="h",
            title=f"{name}: Top Numeric Outlier Percentages",
            labels={"outlier_pct": "Outliers (%)"},
            template=PLOTLY_TEMPLATE,
        )
        fig.show()
        display(Markdown("**Business interpretation:** Outliers can represent true demand spikes, promotions, disruption events, or data errors. They should be reviewed rather than blindly removed."))

    display(Markdown("**Missing value table**"))
    display(report["missing"].head(25))
    if not report["outliers"].empty:
        display(Markdown("**Outlier table**"))
        display(report["outliers"].head(25))


quality_reports = []
for df, name in [
    (forecasting_df, "Forecasting Time Series"),
    (calendar_df, "M5 Calendar"),
    (sales_df, "M5 Sales"),
    (sell_prices_df, "M5 Sell Prices"),
    (dataco_df, "DataCo Supply Chain"),
]:
    report = data_quality_report(df, name)
    quality_reports.append(report)
    visualize_quality_report(report)
'@

Add-Markdown @'
## 6. Exploratory Data Analysis

EDA translates raw data into business understanding. The charts below examine demand trend, seasonality, distributions, category/store/region behavior, shipment patterns, profit behavior, product concentration, and price behavior.

Each visualization includes a short business interpretation so the analysis is review-ready rather than purely technical.
'@

Add-Code @'
# -----------------------------
# Visualization helper functions
# -----------------------------

chart_registry = []

def show_chart(fig, title, interpretation, height=430):
    """Standardize Plotly chart formatting and business interpretation."""
    fig.update_layout(title=title, template=PLOTLY_TEMPLATE, height=height)
    fig.show()
    chart_registry.append(title)
    display(Markdown(f"**Business interpretation:** {interpretation}"))


def show_note(title, message):
    """Display a section note when a chart cannot be generated from available columns."""
    display(Markdown(f"**{title}:** {message}"))


def safe_numeric_col(df, preferred=None, contains_any=None, contains_all=None):
    col = find_column(df, preferred=preferred, contains_any=contains_any, contains_all=contains_all)
    if col is not None:
        return col
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist() if df is not None else []
    return numeric_cols[0] if numeric_cols else None


def aggregate_m5_dimension(sales, dimension, top_n=20):
    """Aggregate M5 demand by a dimensional column such as cat_id, store_id, or state_id."""
    if sales is None or dimension not in sales.columns:
        return pd.DataFrame()
    d_cols = [c for c in sales.columns if re.fullmatch(r"d_\d+", str(c))]
    if not d_cols:
        return pd.DataFrame()
    temp = sales[[dimension] + d_cols].copy()
    temp["total_demand"] = temp[d_cols].sum(axis=1)
    return (
        temp.groupby(dimension, as_index=False)["total_demand"]
        .sum()
        .sort_values("total_demand", ascending=False)
        .head(top_n)
    )


def top_bottom_dataco_products(df, top=True, n=15):
    if df is None:
        return pd.DataFrame()
    product_col = find_column(df, preferred=["Product Name"], contains_all=["product", "name"])
    sales_col = find_column(df, preferred=["Sales", "Order Item Total"], contains_any=["sales", "total"])
    if product_col is None or sales_col is None:
        return pd.DataFrame()
    temp = df[[product_col, sales_col]].copy()
    temp[sales_col] = coerce_numeric(temp[sales_col])
    agg = temp.groupby(product_col, as_index=False)[sales_col].sum()
    return agg.sort_values(sales_col, ascending=not top).head(n)
'@

Add-Code @'
# -----------------------------
# Time-series EDA charts
# -----------------------------

eda_ts = forecasting_df.copy()
eda_ts["Date"] = pd.to_datetime(eda_ts["Date"], errors="coerce")
eda_ts = eda_ts.dropna(subset=["Date"]).sort_values("Date")
eda_ts["Year"] = eda_ts["Date"].dt.year
eda_ts["Month"] = eda_ts["Date"].dt.month
eda_ts["MonthName"] = eda_ts["Date"].dt.strftime("%b")
eda_ts["Week"] = eda_ts["Date"].dt.isocalendar().week.astype(int)
eda_ts["Weekday"] = eda_ts["Date"].dt.day_name()
eda_ts["RollingMean7"] = eda_ts["Demand"].rolling(7).mean()
eda_ts["RollingMean30"] = eda_ts["Demand"].rolling(30).mean()
eda_ts["RollingStd7"] = eda_ts["Demand"].rolling(7).std()
eda_ts["RollingStd30"] = eda_ts["Demand"].rolling(30).std()

# 1. Demand Trend
fig = px.line(eda_ts, x="Date", y="Demand", title="Demand Trend")
show_chart(fig, "Demand Trend", "Long-term direction shows whether planning should prepare for growth, decline, or stable replenishment.")

# 2. Monthly Demand
monthly = eda_ts.set_index("Date")["Demand"].resample("M").sum().reset_index()
fig = px.bar(monthly, x="Date", y="Demand", title="Monthly Demand")
show_chart(fig, "Monthly Demand", "Monthly aggregation reveals planning cycles and recurring volume peaks useful for S&OP meetings.")

# 3. Weekly Demand
weekly = eda_ts.set_index("Date")["Demand"].resample("W").sum().reset_index()
fig = px.line(weekly, x="Date", y="Demand", title="Weekly Demand")
show_chart(fig, "Weekly Demand", "Weekly patterns help align warehouse labor, replenishment cadence, and transportation capacity.")

# 4. Yearly Demand
yearly = eda_ts.groupby("Year", as_index=False)["Demand"].sum()
fig = px.bar(yearly, x="Year", y="Demand", title="Yearly Demand")
show_chart(fig, "Yearly Demand", "Yearly totals show strategic growth or contraction and support annual capacity planning.")

# 5. Histogram
fig = px.histogram(eda_ts, x="Demand", nbins=60, title="Demand Histogram")
show_chart(fig, "Demand Histogram", "The histogram shows whether demand is concentrated, skewed, or volatile.")

# 6. Distribution Plot
fig = px.histogram(eda_ts, x="Demand", nbins=60, marginal="box", histnorm="probability density", title="Demand Distribution")
show_chart(fig, "Demand Distribution Plot", "Distribution shape helps decide whether robust models or transformations are needed.")

# 7. Box Plot by Month
fig = px.box(eda_ts, x="MonthName", y="Demand", points="outliers", title="Demand Box Plot by Month")
show_chart(fig, "Demand Box Plot", "Monthly box plots reveal seasonality and unusual demand periods that require inventory buffers.")

# 8. Correlation Heatmap
corr_cols = ["Demand", "Year", "Month", "Week", "RollingMean7", "RollingMean30", "RollingStd7", "RollingStd30"]
corr = eda_ts[corr_cols].dropna().corr()
fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
show_chart(fig, "Correlation Heatmap", "Correlation highlights which engineered signals move with demand and may support forecasting.")

# 9. Seasonality Analysis by Weekday
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday = eda_ts.groupby("Weekday", as_index=False)["Demand"].mean()
weekday["Weekday"] = pd.Categorical(weekday["Weekday"], categories=weekday_order, ordered=True)
weekday = weekday.sort_values("Weekday")
fig = px.bar(weekday, x="Weekday", y="Demand", title="Average Demand by Weekday")
show_chart(fig, "Seasonality Analysis", "Weekday seasonality supports labor scheduling and order cut-off decisions.")

# 10. Rolling Mean
fig = go.Figure()
fig.add_trace(go.Scatter(x=eda_ts["Date"], y=eda_ts["Demand"], name="Daily Demand", opacity=0.35))
fig.add_trace(go.Scatter(x=eda_ts["Date"], y=eda_ts["RollingMean7"], name="7-Day Rolling Mean"))
fig.add_trace(go.Scatter(x=eda_ts["Date"], y=eda_ts["RollingMean30"], name="30-Day Rolling Mean"))
show_chart(fig, "Rolling Mean", "Rolling means smooth noise and show the signal planners should track for replenishment decisions.")

# 11. Rolling Standard Deviation
fig = go.Figure()
fig.add_trace(go.Scatter(x=eda_ts["Date"], y=eda_ts["RollingStd7"], name="7-Day Rolling Std"))
fig.add_trace(go.Scatter(x=eda_ts["Date"], y=eda_ts["RollingStd30"], name="30-Day Rolling Std"))
show_chart(fig, "Rolling Standard Deviation", "Rising rolling volatility indicates higher safety-stock requirements and forecast uncertainty.")

# 12. Time Series Trend with Decomposition
if STATSMODELS_AVAILABLE and len(eda_ts) >= 730:
    try:
        decomposition = seasonal_decompose(eda_ts.set_index("Date")["Demand"], model="additive", period=365)
        decomp_df = pd.DataFrame({
            "Date": eda_ts["Date"],
            "Observed": decomposition.observed.values,
            "Trend": decomposition.trend.values,
            "Seasonal": decomposition.seasonal.values,
            "Residual": decomposition.resid.values,
        })
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
        for i, col in enumerate(["Observed", "Trend", "Seasonal", "Residual"], start=1):
            fig.add_trace(go.Scatter(x=decomp_df["Date"], y=decomp_df[col], name=col), row=i, col=1)
        show_chart(fig, "Time Series Trend Decomposition", "Decomposition separates trend, seasonality, and residual shock patterns for more explainable planning.", height=850)
    except Exception as exc:
        show_note("Time Series Trend Decomposition", f"Skipped because decomposition failed: {exc}")
else:
    show_note("Time Series Trend Decomposition", "Skipped because statsmodels is unavailable or the series is too short.")
'@

Add-Code @'
# -----------------------------
# Dimensional and supply-chain EDA charts
# -----------------------------

# 13. Product Category Analysis
category_summary = aggregate_m5_dimension(sales_df, "cat_id")
if not category_summary.empty:
    fig = px.bar(category_summary, x="cat_id", y="total_demand", title="Product Category Analysis")
    show_chart(fig, "Product Category Analysis", "Category contribution identifies where inventory policy and demand planning effort should be concentrated.")
else:
    show_note("Product Category Analysis", "M5 category column not available.")

# 14. Store Analysis
store_summary = aggregate_m5_dimension(sales_df, "store_id")
if not store_summary.empty:
    fig = px.bar(store_summary, x="store_id", y="total_demand", title="Store Analysis")
    show_chart(fig, "Store Analysis", "Store-level demand concentration helps prioritize replenishment, allocation, and regional service levels.")
else:
    show_note("Store Analysis", "M5 store column not available.")

# 15. Region / State Analysis
state_summary = aggregate_m5_dimension(sales_df, "state_id")
if not state_summary.empty:
    fig = px.bar(state_summary, x="state_id", y="total_demand", title="Region Analysis")
    show_chart(fig, "Region Analysis", "Regional demand differences point to localized inventory policies rather than one-size-fits-all planning.")
else:
    region_col = find_column(dataco_df, preferred=["Order Region", "Customer Region"], contains_any=["region"])
    sales_col = find_column(dataco_df, preferred=["Sales"], contains_any=["sales"])
    if dataco_df is not None and region_col and sales_col:
        temp = dataco_df[[region_col, sales_col]].copy()
        temp[sales_col] = coerce_numeric(temp[sales_col])
        region_summary = temp.groupby(region_col, as_index=False)[sales_col].sum().sort_values(sales_col, ascending=False).head(20)
        fig = px.bar(region_summary, x=region_col, y=sales_col, title="Region Analysis")
        show_chart(fig, "Region Analysis", "Regional sales exposure helps identify markets requiring stronger logistics and inventory controls.")

# 16. Sales Distribution
sales_col = find_column(dataco_df, preferred=["Sales", "Order Item Total"], contains_any=["sales", "total"])
if dataco_df is not None and sales_col:
    temp = dataco_df[[sales_col]].copy()
    temp[sales_col] = coerce_numeric(temp[sales_col])
    fig = px.histogram(temp.dropna(), x=sales_col, nbins=70, title="Sales Distribution")
    show_chart(fig, "Sales Distribution", "Sales concentration reveals whether revenue depends on many small orders or fewer large orders.")
else:
    show_note("Sales Distribution", "No sales column found in DataCo.")

# 17. Inventory Distribution using order quantity as inventory movement proxy
qty_col = find_column(dataco_df, preferred=["Order Item Quantity", "Quantity"], contains_any=["quantity"])
if dataco_df is not None and qty_col:
    temp = dataco_df[[qty_col]].copy()
    temp[qty_col] = coerce_numeric(temp[qty_col])
    fig = px.histogram(temp.dropna(), x=qty_col, nbins=40, title="Inventory Distribution")
    show_chart(fig, "Inventory Distribution", "Quantity distribution helps estimate demand granularity and replenishment lot-size behavior.")
else:
    show_note("Inventory Distribution", "No quantity column found in DataCo.")

# 18. Shipment Distribution
ship_days_col = find_column(dataco_df, preferred=["Days for shipping (real)"], contains_all=["days", "shipping"])
if dataco_df is not None and ship_days_col:
    temp = dataco_df[[ship_days_col]].copy()
    temp[ship_days_col] = coerce_numeric(temp[ship_days_col])
    fig = px.histogram(temp.dropna(), x=ship_days_col, nbins=30, title="Shipment Distribution")
    show_chart(fig, "Shipment Distribution", "Shipment-time distribution reveals service variability and late-delivery risk.")
else:
    show_note("Shipment Distribution", "No shipment duration column found in DataCo.")

# 19. Profit Distribution
profit_col = find_column(dataco_df, preferred=["Benefit per order", "Order Profit Per Order"], contains_any=["profit", "benefit"])
if dataco_df is not None and profit_col:
    temp = dataco_df[[profit_col]].copy()
    temp[profit_col] = coerce_numeric(temp[profit_col])
    fig = px.histogram(temp.dropna(), x=profit_col, nbins=70, title="Profit Distribution")
    show_chart(fig, "Profit Distribution", "Profit distribution highlights margin risk, loss-making orders, and cost leakage.")
else:
    show_note("Profit Distribution", "No profit column found in DataCo.")

# 20. Top Products
top_products = top_bottom_dataco_products(dataco_df, top=True)
if not top_products.empty:
    product_col = top_products.columns[0]
    value_col = top_products.columns[1]
    fig = px.bar(top_products, x=value_col, y=product_col, orientation="h", title="Top Products")
    show_chart(fig, "Top Products", "Top products show revenue concentration and where forecast accuracy has the highest financial impact.")
else:
    show_note("Top Products", "No product/sales columns found in DataCo.")

# 21. Bottom Products
bottom_products = top_bottom_dataco_products(dataco_df, top=False)
if not bottom_products.empty:
    product_col = bottom_products.columns[0]
    value_col = bottom_products.columns[1]
    fig = px.bar(bottom_products, x=value_col, y=product_col, orientation="h", title="Bottom Products")
    show_chart(fig, "Bottom Products", "Bottom products may represent long-tail SKUs that need different inventory and replenishment policies.")
else:
    show_note("Bottom Products", "No product/sales columns found in DataCo.")

# 22. Late Deliveries
late_col = find_column(dataco_df, preferred=["Late_delivery_risk"], contains_all=["late", "risk"])
delivery_status_col = find_column(dataco_df, preferred=["Delivery Status"], contains_all=["delivery", "status"])
if dataco_df is not None and late_col:
    late_counts = dataco_df[late_col].value_counts(dropna=False).rename_axis("late_delivery_risk").reset_index(name="orders")
    fig = px.bar(late_counts, x="late_delivery_risk", y="orders", title="Late Deliveries")
    show_chart(fig, "Late Deliveries", "Late-delivery exposure directly affects customer experience and logistics SLA performance.")
elif dataco_df is not None and delivery_status_col:
    status_counts = dataco_df[delivery_status_col].value_counts(dropna=False).rename_axis("delivery_status").reset_index(name="orders")
    fig = px.bar(status_counts, x="delivery_status", y="orders", title="Late Deliveries by Status")
    show_chart(fig, "Late Deliveries", "Delivery status distribution reveals operational reliability and exception volume.")
else:
    show_note("Late Deliveries", "No late-delivery indicator found in DataCo.")

# 23. Top Markets
market_col = find_column(dataco_df, preferred=["Market"], contains_any=["market"])
if dataco_df is not None and market_col and sales_col:
    temp = dataco_df[[market_col, sales_col]].copy()
    temp[sales_col] = coerce_numeric(temp[sales_col])
    market_summary = temp.groupby(market_col, as_index=False)[sales_col].sum().sort_values(sales_col, ascending=False)
    fig = px.bar(market_summary, x=market_col, y=sales_col, title="Top Markets")
    show_chart(fig, "Top Markets", "Market concentration supports regional network design and logistics capacity decisions.")
else:
    show_note("Top Markets", "No market/sales columns found in DataCo.")

# 24. Price Distribution
if sell_prices_df is not None and "sell_price" in sell_prices_df.columns:
    fig = px.histogram(sell_prices_df, x="sell_price", nbins=70, title="Sell Price Distribution")
    show_chart(fig, "Sell Price Distribution", "Price spread helps explain demand elasticity and promotion sensitivity.")
else:
    show_note("Sell Price Distribution", "No sell_price column found in sell_prices.")

display(Markdown(f"### EDA chart count generated: {len(chart_registry)}"))
'@

Add-Markdown @'
## 7. Feature Engineering

Forecasting models need predictive signals. This section creates:

- **Lag features:** previous demand values at 1, 7, 14, and 30 days.
- **Rolling features:** moving averages and moving volatility over 7 and 30 days.
- **Calendar features:** year, month, quarter, ISO week, weekday, weekend flag.
- **Trend features:** numeric time index for long-term movement.
- **Seasonality features:** sine/cosine encodings for cyclical month, week, and day-of-year effects.

These features are production-friendly because they can be generated from timestamp and historical demand during batch or real-time scoring.
'@

Add-Code @'
def create_time_series_features(df, date_col="Date", target_col="Demand"):
    """Create reusable forecasting features from a univariate daily time series."""
    feature_df = df[[date_col, target_col]].copy()
    feature_df[date_col] = pd.to_datetime(feature_df[date_col], errors="coerce")
    feature_df[target_col] = coerce_numeric(feature_df[target_col])
    feature_df = feature_df.dropna(subset=[date_col, target_col]).sort_values(date_col).reset_index(drop=True)

    # Lag features capture autocorrelation and delayed demand effects.
    for lag in [1, 7, 14, 30]:
        feature_df[f"Lag{lag}"] = feature_df[target_col].shift(lag)

    # Rolling features summarize recent baseline and volatility.
    feature_df["RollingMean7"] = feature_df[target_col].shift(1).rolling(window=7).mean()
    feature_df["RollingMean30"] = feature_df[target_col].shift(1).rolling(window=30).mean()
    feature_df["RollingStd7"] = feature_df[target_col].shift(1).rolling(window=7).std()
    feature_df["RollingStd30"] = feature_df[target_col].shift(1).rolling(window=30).std()

    # Calendar features capture seasonality and operational rhythms.
    dt = feature_df[date_col].dt
    iso = dt.isocalendar()
    feature_df["Year"] = dt.year
    feature_df["Month"] = dt.month
    feature_df["Quarter"] = dt.quarter
    feature_df["Week"] = iso.week.astype(int)
    feature_df["Weekday"] = dt.weekday
    feature_df["IsWeekend"] = feature_df["Weekday"].isin([5, 6]).astype(int)
    feature_df["DayOfYear"] = dt.dayofyear

    # Trend feature gives models a simple long-run index.
    feature_df["Trend"] = np.arange(len(feature_df))

    # Cyclical encodings prevent artificial discontinuities between period boundaries.
    feature_df["MonthSin"] = np.sin(2 * np.pi * feature_df["Month"] / 12)
    feature_df["MonthCos"] = np.cos(2 * np.pi * feature_df["Month"] / 12)
    feature_df["WeekSin"] = np.sin(2 * np.pi * feature_df["Week"] / 52)
    feature_df["WeekCos"] = np.cos(2 * np.pi * feature_df["Week"] / 52)
    feature_df["DayOfYearSin"] = np.sin(2 * np.pi * feature_df["DayOfYear"] / 365.25)
    feature_df["DayOfYearCos"] = np.cos(2 * np.pi * feature_df["DayOfYear"] / 365.25)

    return feature_df


feature_df = create_time_series_features(forecasting_df)
model_df = feature_df.dropna().reset_index(drop=True)

feature_cols = [
    c for c in model_df.columns
    if c not in ["Date", "Demand"] and pd.api.types.is_numeric_dtype(model_df[c])
]

display(Markdown("### Engineered Feature Preview"))
display(model_df.head())
print(f"Modeling rows after lag/rolling feature drop: {len(model_df):,}")
print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
'@

Add-Markdown @'
## 8. Forecasting Models

This section implements statistical, machine learning, and deep learning forecasting models.

### 1. ARIMA

**Theory:** ARIMA models demand using autoregression, differencing, and moving-average components. It is effective for time series where current values depend on past values and past errors.

**Advantages:** Interpretable, strong classical baseline, useful for stable aggregate demand.

**Disadvantages:** Limited nonlinear modeling, requires stationarity assumptions, struggles with complex external drivers.

**Business use cases:** Aggregate demand planning, mature products, stable replenishment environments.

### 2. SARIMA

**Theory:** SARIMA extends ARIMA with seasonal autoregressive and moving-average terms.

**Advantages:** Captures weekly or yearly seasonality, interpretable, strong benchmark for recurring demand cycles.

**Disadvantages:** Can be slow on long series, requires seasonal period selection, limited nonlinear behavior.

**Business use cases:** Seasonal retail demand, weekly replenishment cycles, recurring promotion calendars.

### 3. Random Forest

**Theory:** Random Forest averages many decision trees trained on bootstrapped samples and feature subsets.

**Advantages:** Handles nonlinear relationships, robust to outliers, low feature scaling requirements.

**Disadvantages:** Less extrapolative beyond observed patterns, larger models can be less interpretable.

**Business use cases:** SKU/store forecasting with lag, calendar, promotion, price, and operational features.

### 4. XGBoost

**Theory:** XGBoost builds boosted decision trees sequentially, where each tree corrects previous errors.

**Advantages:** High accuracy, strong nonlinear modeling, handles mixed features and interactions well.

**Disadvantages:** Requires tuning, can overfit, less transparent than classical models.

**Business use cases:** High-value SKU forecasting, demand sensing, feature-rich planning systems.

### 5. LightGBM

**Theory:** LightGBM is a gradient boosting framework optimized for speed and large datasets using histogram-based tree learning.

**Advantages:** Fast training, scalable, strong accuracy for tabular forecasting features.

**Disadvantages:** Sensitive to parameter choices, can overfit small datasets.

**Business use cases:** Large-scale enterprise forecasting across many products, stores, and regions.

### 6. LSTM

**Theory:** Long Short-Term Memory networks are recurrent neural networks designed to learn long-range sequence dependencies.

**Advantages:** Captures temporal patterns, nonlinear dynamics, and sequential memory.

**Disadvantages:** Requires more data and compute, less interpretable, tuning sensitive.

**Business use cases:** Complex demand sequences with long memory, multivariate time-series extensions.

### 7. GRU

**Theory:** Gated Recurrent Units are recurrent neural networks with fewer gates than LSTM, often training faster while retaining sequence memory.

**Advantages:** Efficient, sequence-aware, sometimes performs similarly to LSTM with less complexity.

**Disadvantages:** Requires deep learning stack and tuning; not always superior to tree models for tabular features.

**Business use cases:** Faster neural forecasting prototypes and streaming demand sequence modeling.
'@

Add-Code @'
# -----------------------------
# Modeling utilities
# -----------------------------

def temporal_train_test_split(df, test_size=TEST_SIZE_DAYS):
    """Chronological train-test split for time series."""
    if len(df) < 120:
        raise ValueError("Not enough rows for robust time-series modeling.")
    test_size = min(test_size, max(14, int(len(df) * 0.2)))
    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    return train, test


def evaluate_forecast(y_true, y_pred):
    """Compute standard forecast metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    min_len = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:min_len], y_pred[:min_len]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


model_results = []
model_artifacts = {}
prediction_frames = {}


def register_model_result(name, y_true, y_pred, dates, model_object=None, model_type="unknown", metadata=None):
    """Register metrics, predictions, and trained model artifacts."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    min_len = min(len(y_true), len(y_pred), len(dates))

    metrics = evaluate_forecast(y_true[:min_len], y_pred[:min_len])
    row = {"Model": name, **metrics}
    model_results.append(row)

    pred_df = pd.DataFrame({
        "Date": pd.to_datetime(pd.Series(dates).iloc[:min_len].values),
        "Actual": y_true[:min_len],
        "Predicted": y_pred[:min_len],
        "Residual": y_true[:min_len] - y_pred[:min_len],
    })
    prediction_frames[name] = pred_df
    model_artifacts[name] = {
        "model": model_object,
        "type": model_type,
        "metadata": metadata or {},
        "metrics": metrics,
    }
    print(f"Registered {name}: RMSE={metrics['RMSE']:,.2f}, MAE={metrics['MAE']:,.2f}, MAPE={metrics['MAPE']:,.2f}%, R2={metrics['R2']:,.4f}")


train_df, test_df = temporal_train_test_split(model_df, TEST_SIZE_DAYS)
X_train, y_train = train_df[feature_cols], train_df["Demand"]
X_test, y_test = test_df[feature_cols], test_df["Demand"]

display(Markdown(f"**Train rows:** {len(train_df):,} | **Test rows:** {len(test_df):,}"))
'@

Add-Code @'
# -----------------------------
# Statistical models: ARIMA and SARIMA
# -----------------------------

def train_arima_model(train_series, test_len, order=(2, 1, 2)):
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is not available.")
    model = SARIMAX(
        train_series,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=100)
    forecast = fitted.forecast(steps=test_len)
    return fitted, forecast


def train_sarima_model(train_series, test_len, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)):
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is not available.")
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=100)
    forecast = fitted.forecast(steps=test_len)
    return fitted, forecast


for model_name, trainer, metadata in [
    ("ARIMA", train_arima_model, {"order": (2, 1, 2)}),
    ("SARIMA", train_sarima_model, {"order": (1, 1, 1), "seasonal_order": (1, 0, 1, 7)}),
]:
    try:
        fitted_model, preds = trainer(y_train, len(test_df))
        register_model_result(
            model_name,
            y_test.values,
            preds,
            test_df["Date"].values,
            model_object=fitted_model,
            model_type="statistical",
            metadata=metadata,
        )
    except Exception as exc:
        print(f"{model_name} skipped: {exc}")
'@

Add-Code @'
# -----------------------------
# Machine learning models: Random Forest, XGBoost, LightGBM
# -----------------------------

ml_models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
}

if XGBOOST_AVAILABLE:
    ml_models["XGBoost"] = XGBRegressor(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

if LIGHTGBM_AVAILABLE:
    ml_models["LightGBM"] = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        verbose=-1,
    )

for model_name, model in ml_models.items():
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        register_model_result(
            model_name,
            y_test.values,
            preds,
            test_df["Date"].values,
            model_object=model,
            model_type="machine_learning",
            metadata={"feature_cols": feature_cols},
        )
    except Exception as exc:
        print(f"{model_name} skipped: {exc}")
'@

Add-Code @'
# -----------------------------
# Deep learning models: LSTM and GRU
# -----------------------------

def create_sequences(values, sequence_length):
    """Create supervised sequences for recurrent neural networks."""
    X, y, target_indices = [], [], []
    for i in range(sequence_length, len(values)):
        X.append(values[i - sequence_length:i])
        y.append(values[i])
        target_indices.append(i)
    return np.array(X), np.array(y), np.array(target_indices)


def build_recurrent_model(kind="LSTM", sequence_length=SEQUENCE_LENGTH):
    """Build a compact recurrent model for daily demand forecasting."""
    layer = LSTM if kind.upper() == "LSTM" else GRU
    model = Sequential([
        layer(64, input_shape=(sequence_length, 1), return_sequences=False),
        Dropout(0.15),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_deep_sequence_model(kind, full_model_df, split_index):
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not available.")
    if len(full_model_df) < SEQUENCE_LENGTH + 90:
        raise ValueError("Not enough rows for recurrent model training.")

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(full_model_df[["Demand"]]).reshape(-1, 1)

    X_seq, y_seq, target_indices = create_sequences(scaled_values, SEQUENCE_LENGTH)
    train_mask = target_indices < split_index
    test_mask = target_indices >= split_index

    X_train_seq = X_seq[train_mask].reshape((-1, SEQUENCE_LENGTH, 1))
    y_train_seq = y_seq[train_mask].reshape((-1, 1))
    X_test_seq = X_seq[test_mask].reshape((-1, SEQUENCE_LENGTH, 1))
    y_test_seq = y_seq[test_mask].reshape((-1, 1))
    test_indices = target_indices[test_mask]

    model = build_recurrent_model(kind, SEQUENCE_LENGTH)
    callbacks = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    model.fit(
        X_train_seq,
        y_train_seq,
        validation_split=0.1,
        epochs=DEEP_LEARNING_EPOCHS,
        batch_size=DEEP_LEARNING_BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )

    pred_scaled = model.predict(X_test_seq, verbose=0)
    preds = scaler.inverse_transform(pred_scaled).reshape(-1)
    actuals = scaler.inverse_transform(y_test_seq).reshape(-1)
    dates = full_model_df["Date"].iloc[test_indices].values
    return model, scaler, actuals, preds, dates


for kind in ["LSTM", "GRU"]:
    try:
        deep_model, deep_scaler, actuals, preds, dates = train_deep_sequence_model(kind, model_df, len(train_df))
        register_model_result(
            kind,
            actuals,
            preds,
            dates,
            model_object=deep_model,
            model_type="deep_learning",
            metadata={"scaler": deep_scaler, "sequence_length": SEQUENCE_LENGTH},
        )
    except Exception as exc:
        print(f"{kind} skipped: {exc}")
'@

Add-Markdown @'
## 9. Model Evaluation

Forecasting models are evaluated using:

- **MAE:** Average absolute forecast error.
- **RMSE:** Penalizes large errors more strongly and is used for best-model selection.
- **MAPE:** Percentage error, useful for business interpretation when demand values are non-zero.
- **R2 Score:** Explains variance captured by the model.
'@

Add-Code @'
if not model_results:
    raise ValueError("No forecasting models were successfully trained. Check dependencies and data quality.")

metrics_df = pd.DataFrame(model_results)
metrics_df = metrics_df[["Model", "MAE", "RMSE", "MAPE", "R2"]].sort_values("RMSE").reset_index(drop=True)
metrics_df["Rank"] = np.arange(1, len(metrics_df) + 1)

display(Markdown("### Model Evaluation Metrics"))
display(metrics_df.style.format({
    "MAE": "{:,.2f}",
    "RMSE": "{:,.2f}",
    "MAPE": "{:,.2f}",
    "R2": "{:,.4f}",
}))

for model_name, pred_df in prediction_frames.items():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Actual"], name="Actual"))
    fig.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Predicted"], name="Predicted"))
    show_chart(
        fig,
        f"{model_name}: Actual vs Predicted",
        "Forecast tracking shows whether the model is directionally aligned with actual demand and where residual risk remains."
    )
'@

Add-Markdown @'
## 10. Model Comparison

The comparison below ranks models automatically and visualizes metric trade-offs. In production, the final model choice should consider not only accuracy but also latency, interpretability, retraining cost, operational risk, and explainability.
'@

Add-Code @'
def plot_metric_comparison(metrics, metric, lower_is_better=True):
    sorted_metrics = metrics.sort_values(metric, ascending=lower_is_better)
    fig = px.bar(
        sorted_metrics,
        x="Model",
        y=metric,
        color="Model",
        title=f"{metric} Comparison",
        text=metric,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    show_chart(
        fig,
        f"{metric} Comparison",
        f"{metric} comparison helps evaluate relative forecast performance across statistical, ML, and deep learning approaches."
    )


plot_metric_comparison(metrics_df, "RMSE", lower_is_better=True)
plot_metric_comparison(metrics_df, "MAE", lower_is_better=True)
plot_metric_comparison(metrics_df, "MAPE", lower_is_better=True)
plot_metric_comparison(metrics_df, "R2", lower_is_better=False)

display(Markdown("### Detailed Comparison Table"))
display(metrics_df)


def generate_comparison_narrative(metrics):
    best = metrics.iloc[0]
    worst = metrics.sort_values("RMSE", ascending=False).iloc[0]
    model_type = model_artifacts.get(best["Model"], {}).get("type", "unknown")

    if model_type == "machine_learning":
        reason = (
            "The best model is a tree-based machine learning model. This usually means lag, rolling, "
            "trend, and calendar features contain nonlinear interactions that classical models do not capture as well."
        )
    elif model_type == "statistical":
        reason = (
            "The best model is statistical. This suggests the aggregate demand series has strong autocorrelation "
            "and relatively stable temporal structure."
        )
    elif model_type == "deep_learning":
        reason = (
            "The best model is recurrent deep learning. This suggests sequential dependencies are important, "
            "although production use should still evaluate explainability and retraining cost."
        )
    else:
        reason = "The best model achieved the lowest RMSE on the holdout period."

    return f"""
### Model Comparison Interpretation

Best model by RMSE: **{best['Model']}** with RMSE **{best['RMSE']:,.2f}**.

Weakest model by RMSE: **{worst['Model']}** with RMSE **{worst['RMSE']:,.2f}**.

{reason}

From a business perspective, the lowest-RMSE model is preferred because large demand errors cause the most expensive planning failures: stockouts, expedited shipments, excess inventory, and missed service levels.
"""


display(Markdown(generate_comparison_narrative(metrics_df)))
'@

Add-Markdown @'
## 11. Best Model Selection

The best model is selected automatically using **lowest RMSE**, because RMSE penalizes large errors that can create costly stockouts or overstocking.
'@

Add-Code @'
best_row = metrics_df.sort_values("RMSE").iloc[0]
best_model_name = best_row["Model"]
best_artifact = model_artifacts[best_model_name]

display(Markdown(f"""
### Best Model

**Selected model:** {best_model_name}

**Selection reason:** It achieved the lowest RMSE on the chronological holdout set.

**Performance metrics**

- MAE: {best_row['MAE']:,.2f}
- RMSE: {best_row['RMSE']:,.2f}
- MAPE: {best_row['MAPE']:,.2f}%
- R2: {best_row['R2']:,.4f}
"""))
'@

Add-Markdown @'
## 12. Future Forecasting

Using the best model, this section forecasts future demand for:

- 30 days
- 60 days
- 90 days

The chart overlays historical demand and the future forecast so planners can translate the forecast into replenishment, procurement, labor, and transportation decisions.
'@

Add-Code @'
def build_future_feature_row(history_df, next_date, feature_cols):
    """Build one future feature row using only information available up to the forecast date."""
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

    future_X = pd.DataFrame([row])
    for col in feature_cols:
        if col not in future_X.columns:
            future_X[col] = 0
    return future_X[feature_cols].fillna(method="ffill").fillna(0)


def new_ml_model(model_name):
    """Create a fresh model for full-history refit before future forecasting."""
    if model_name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, max_depth=16, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
    if model_name == "XGBoost" and XGBOOST_AVAILABLE:
        return XGBRegressor(n_estimators=400, learning_rate=0.04, max_depth=5, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1)
    if model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
        return LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, verbose=-1)
    return None


def seasonal_naive_future(history_df, horizon, season=7):
    """Reliable fallback forecast when a selected model cannot be refit for future steps."""
    values = history_df["Demand"].astype(float).values
    future = []
    for i in range(horizon):
        if len(values) >= season:
            pred = values[-season + (i % season)]
        else:
            pred = np.mean(values)
        future.append(max(pred, 0))
    return np.array(future)


def forecast_future(best_model_name, history_df, horizon=90):
    """Forecast future demand using the selected model where possible."""
    history = history_df[["Date", "Demand"]].copy().sort_values("Date").reset_index(drop=True)
    full_features = create_time_series_features(history).dropna().reset_index(drop=True)
    last_date = history["Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    artifact_type = model_artifacts.get(best_model_name, {}).get("type")

    try:
        if artifact_type == "machine_learning":
            model = new_ml_model(best_model_name)
            if model is None:
                raise ValueError("No constructor available for selected ML model.")
            model.fit(full_features[feature_cols], full_features["Demand"])

            working_history = history.copy()
            preds = []
            for next_date in future_dates:
                X_next = build_future_feature_row(working_history, next_date, feature_cols)
                pred = float(np.clip(model.predict(X_next)[0], 0, None))
                preds.append(pred)
                working_history = pd.concat([
                    working_history,
                    pd.DataFrame({"Date": [next_date], "Demand": [pred]})
                ], ignore_index=True)

        elif artifact_type == "statistical" and STATSMODELS_AVAILABLE:
            if best_model_name == "SARIMA":
                model = SARIMAX(history["Demand"], order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
            else:
                model = SARIMAX(history["Demand"], order=(2, 1, 2), enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit(disp=False, maxiter=100)
            preds = np.clip(fitted.forecast(steps=horizon).values, 0, None)

        elif artifact_type == "deep_learning" and TENSORFLOW_AVAILABLE:
            model = model_artifacts[best_model_name]["model"]
            scaler = model_artifacts[best_model_name]["metadata"]["scaler"]
            scaled_history = scaler.transform(history[["Demand"]]).reshape(-1).tolist()
            preds_scaled = []
            for _ in range(horizon):
                seq = np.array(scaled_history[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, 1)
                pred_scaled = float(model.predict(seq, verbose=0)[0][0])
                preds_scaled.append(pred_scaled)
                scaled_history.append(pred_scaled)
            preds = np.clip(scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).reshape(-1), 0, None)

        else:
            preds = seasonal_naive_future(history, horizon)

    except Exception as exc:
        print(f"Future forecasting with {best_model_name} failed, using seasonal naive fallback: {exc}")
        preds = seasonal_naive_future(history, horizon)

    return pd.DataFrame({"Date": future_dates, "ForecastDemand": preds})


future_forecast_df = forecast_future(best_model_name, forecasting_df, horizon=max(FORECAST_HORIZONS))
display(future_forecast_df.head())
display(future_forecast_df.tail())

for horizon in FORECAST_HORIZONS:
    horizon_sum = future_forecast_df.head(horizon)["ForecastDemand"].sum()
    display(Markdown(f"**{horizon}-day forecast demand:** {horizon_sum:,.0f} units"))

historical_window = forecasting_df.sort_values("Date").tail(365)
fig = go.Figure()
fig.add_trace(go.Scatter(x=historical_window["Date"], y=historical_window["Demand"], name="Historical Demand"))
fig.add_trace(go.Scatter(x=future_forecast_df["Date"], y=future_forecast_df["ForecastDemand"], name="Forecast Demand"))
show_chart(
    fig,
    "Historical Demand vs 90-Day Forecast",
    "The forecast horizon supports procurement timing, stock allocation, capacity planning, and proactive exception management."
)
'@

Add-Markdown @'
## 13. Anomaly Detection

Supply chains need early warning systems. This section uses **Isolation Forest** to detect:

- Demand spikes
- Demand drops
- Inventory or order-quantity anomalies

Anomalies are not automatically bad data. They may represent promotions, disruptions, stockouts, weather events, supplier issues, or logistics constraints.
'@

Add-Code @'
def detect_demand_anomalies(feature_data, contamination=ANOMALY_CONTAMINATION):
    anomaly_cols = ["Demand", "Lag1", "Lag7", "Lag14", "Lag30", "RollingMean7", "RollingMean30", "RollingStd7", "RollingStd30"]
    available_cols = [c for c in anomaly_cols if c in feature_data.columns]
    anomaly_df = feature_data.dropna(subset=available_cols).copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(anomaly_df[available_cols])
    detector = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    anomaly_df["AnomalyFlag"] = detector.fit_predict(X)
    anomaly_df["IsAnomaly"] = anomaly_df["AnomalyFlag"].eq(-1)
    anomaly_df["AnomalyType"] = np.where(
        anomaly_df["Demand"] >= anomaly_df["RollingMean30"],
        "Demand Spike",
        "Demand Drop"
    )
    return anomaly_df, detector


demand_anomaly_df, demand_anomaly_model = detect_demand_anomalies(feature_df)
demand_anomalies = demand_anomaly_df[demand_anomaly_df["IsAnomaly"]].copy()

display(Markdown(f"### Demand Anomalies Detected: {len(demand_anomalies):,}"))
display(demand_anomalies[["Date", "Demand", "RollingMean30", "AnomalyType"]].head(20))

fig = go.Figure()
fig.add_trace(go.Scatter(x=demand_anomaly_df["Date"], y=demand_anomaly_df["Demand"], name="Demand", opacity=0.55))
fig.add_trace(go.Scatter(
    x=demand_anomalies["Date"],
    y=demand_anomalies["Demand"],
    mode="markers",
    name="Anomaly",
    marker=dict(color="red", size=8)
))
show_chart(
    fig,
    "Demand Anomaly Detection",
    "Demand anomalies signal periods where standard replenishment rules may fail and planners should investigate root causes."
)

anomaly_counts = demand_anomalies["AnomalyType"].value_counts().reset_index()
anomaly_counts.columns = ["AnomalyType", "Count"]
fig = px.bar(anomaly_counts, x="AnomalyType", y="Count", title="Demand Spikes vs Demand Drops")
show_chart(
    fig,
    "Demand Spike and Drop Counts",
    "Separating spikes from drops helps distinguish stockout risk from overstock or demand-loss risk."
)


def detect_inventory_anomalies(df, contamination=ANOMALY_CONTAMINATION):
    if df is None:
        return pd.DataFrame(), None
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
        return pd.DataFrame(), None

    temp = df[candidate_cols].copy()
    for col in candidate_cols:
        temp[col] = coerce_numeric(temp[col])
    temp = temp.dropna()
    if len(temp) < 100:
        return pd.DataFrame(), None

    scaler = StandardScaler()
    X = scaler.fit_transform(temp)
    detector = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    flags = detector.fit_predict(X)
    result = df.loc[temp.index].copy()
    result["InventoryAnomaly"] = flags == -1
    return result[result["InventoryAnomaly"]].copy(), detector


inventory_anomalies, inventory_anomaly_model = detect_inventory_anomalies(dataco_df)
display(Markdown(f"### Inventory / Order Anomalies Detected: {len(inventory_anomalies):,}"))
display(inventory_anomalies.head(10))
'@

Add-Markdown @'
## 14. Supply Chain Intelligence

This section uses the DataCo dataset to generate operational intelligence:

- Inventory KPIs
- Order KPIs
- Shipment KPIs
- Profit KPIs
- Risk KPIs

The goal is to move from raw operational records to executive-ready decision signals.
'@

Add-Code @'
def compute_supply_chain_kpis(df):
    if df is None or df.empty:
        return {}, {}

    cols = {
        "order_id": find_column(df, preferred=["Order Id", "Order ID"], contains_all=["order", "id"]),
        "sales": find_column(df, preferred=["Sales", "Order Item Total"], contains_any=["sales", "total"]),
        "quantity": find_column(df, preferred=["Order Item Quantity", "Quantity"], contains_any=["quantity"]),
        "profit": find_column(df, preferred=["Benefit per order", "Order Profit Per Order"], contains_any=["profit", "benefit"]),
        "late_risk": find_column(df, preferred=["Late_delivery_risk"], contains_all=["late", "risk"]),
        "delivery_status": find_column(df, preferred=["Delivery Status"], contains_all=["delivery", "status"]),
        "real_days": find_column(df, preferred=["Days for shipping (real)"], contains_all=["days", "shipping"]),
        "scheduled_days": find_column(df, preferred=["Days for shipment (scheduled)"], contains_all=["days", "scheduled"]),
        "region": find_column(df, preferred=["Order Region"], contains_any=["region"]),
        "market": find_column(df, preferred=["Market"], contains_any=["market"]),
        "product": find_column(df, preferred=["Product Name"], contains_all=["product", "name"]),
        "shipping_mode": find_column(df, preferred=["Shipping Mode"], contains_all=["shipping", "mode"]),
    }

    working = df.copy()
    for key in ["sales", "quantity", "profit", "late_risk", "real_days", "scheduled_days"]:
        if cols[key]:
            working[cols[key]] = coerce_numeric(working[cols[key]])

    total_orders = working[cols["order_id"]].nunique() if cols["order_id"] else len(working)
    total_sales = working[cols["sales"]].sum() if cols["sales"] else np.nan
    total_units = working[cols["quantity"]].sum() if cols["quantity"] else np.nan
    total_profit = working[cols["profit"]].sum() if cols["profit"] else np.nan
    avg_order_value = total_sales / total_orders if cols["sales"] and total_orders else np.nan
    profit_margin = total_profit / total_sales if cols["profit"] and cols["sales"] and total_sales else np.nan

    if cols["late_risk"]:
        late_rate = working[cols["late_risk"]].mean()
    elif cols["delivery_status"]:
        late_rate = working[cols["delivery_status"]].astype(str).str.lower().str.contains("late").mean()
    else:
        late_rate = np.nan

    if cols["real_days"] and cols["scheduled_days"]:
        delay_days = working[cols["real_days"]] - working[cols["scheduled_days"]]
        avg_delay = delay_days.mean()
        delay_rate = (delay_days > 0).mean()
    else:
        avg_delay = np.nan
        delay_rate = np.nan

    negative_profit_rate = (working[cols["profit"]] < 0).mean() if cols["profit"] else np.nan

    kpis = {
        "Total Orders": total_orders,
        "Total Sales": total_sales,
        "Total Units": total_units,
        "Average Order Value": avg_order_value,
        "Total Profit": total_profit,
        "Profit Margin": profit_margin,
        "Late Delivery Rate": late_rate,
        "Average Delay Days": avg_delay,
        "Shipment Delay Rate": delay_rate,
        "Negative Profit Rate": negative_profit_rate,
    }
    return kpis, cols


dataco_kpis, dataco_cols = compute_supply_chain_kpis(dataco_df)
kpi_table = pd.DataFrame([{"KPI": k, "Value": v} for k, v in dataco_kpis.items()])
display(Markdown("### Supply Chain KPI Table"))
display(kpi_table)

def format_kpi_value(value, pct=False):
    if pd.isna(value):
        return "N/A"
    return f"{value:.2%}" if pct else f"{value:,.2f}"

display(Markdown(f"""
### KPI Interpretation

- **Inventory KPI:** Total units moved = {format_kpi_value(dataco_kpis.get('Total Units'))}. This approximates inventory movement and replenishment load.
- **Order KPI:** Total orders = {format_kpi_value(dataco_kpis.get('Total Orders'))}; average order value = {format_kpi_value(dataco_kpis.get('Average Order Value'))}.
- **Shipment KPI:** Late delivery rate = {format_kpi_value(dataco_kpis.get('Late Delivery Rate'), pct=True)}; average delay days = {format_kpi_value(dataco_kpis.get('Average Delay Days'))}.
- **Profit KPI:** Total profit = {format_kpi_value(dataco_kpis.get('Total Profit'))}; profit margin = {format_kpi_value(dataco_kpis.get('Profit Margin'), pct=True)}.
- **Risk KPI:** Negative profit rate = {format_kpi_value(dataco_kpis.get('Negative Profit Rate'), pct=True)}.
"""))
'@

Add-Code @'
# -----------------------------
# Supply chain intelligence visualizations
# -----------------------------

if dataco_df is not None and dataco_cols:
    # Late Deliveries
    if dataco_cols.get("late_risk"):
        late_counts = dataco_df[dataco_cols["late_risk"]].value_counts(dropna=False).rename_axis("Late Delivery Risk").reset_index(name="Orders")
        fig = px.pie(late_counts, names="Late Delivery Risk", values="Orders", title="Late Deliveries")
        show_chart(fig, "Late Deliveries", "A high late-delivery share indicates logistics reliability risk and potential customer churn.")

    # Profit by Region
    if dataco_cols.get("region") and dataco_cols.get("profit"):
        temp = dataco_df[[dataco_cols["region"], dataco_cols["profit"]]].copy()
        temp[dataco_cols["profit"]] = coerce_numeric(temp[dataco_cols["profit"]])
        region_profit = temp.groupby(dataco_cols["region"], as_index=False)[dataco_cols["profit"]].sum().sort_values(dataco_cols["profit"], ascending=False).head(20)
        fig = px.bar(region_profit, x=dataco_cols["region"], y=dataco_cols["profit"], title="Profit by Region")
        show_chart(fig, "Profit by Region", "Regional profitability identifies where service cost, pricing, and fulfillment strategy need adjustment.")

    # Top Products
    if dataco_cols.get("product") and dataco_cols.get("sales"):
        temp = dataco_df[[dataco_cols["product"], dataco_cols["sales"]]].copy()
        temp[dataco_cols["sales"]] = coerce_numeric(temp[dataco_cols["sales"]])
        product_sales = temp.groupby(dataco_cols["product"], as_index=False)[dataco_cols["sales"]].sum().sort_values(dataco_cols["sales"], ascending=False).head(20)
        fig = px.bar(product_sales, x=dataco_cols["sales"], y=dataco_cols["product"], orientation="h", title="Top Products")
        show_chart(fig, "Top Products by Sales", "High-sales products deserve tighter forecast governance and service-level monitoring.")

    # Top Markets
    if dataco_cols.get("market") and dataco_cols.get("sales"):
        temp = dataco_df[[dataco_cols["market"], dataco_cols["sales"]]].copy()
        temp[dataco_cols["sales"]] = coerce_numeric(temp[dataco_cols["sales"]])
        market_sales = temp.groupby(dataco_cols["market"], as_index=False)[dataco_cols["sales"]].sum().sort_values(dataco_cols["sales"], ascending=False)
        fig = px.bar(market_sales, x=dataco_cols["market"], y=dataco_cols["sales"], title="Top Markets")
        show_chart(fig, "Top Markets", "Market-level concentration informs regional inventory positioning and transport network design.")

    # Shipment Delays
    if dataco_cols.get("real_days") and dataco_cols.get("scheduled_days") and dataco_cols.get("shipping_mode"):
        temp = dataco_df[[dataco_cols["shipping_mode"], dataco_cols["real_days"], dataco_cols["scheduled_days"]]].copy()
        temp[dataco_cols["real_days"]] = coerce_numeric(temp[dataco_cols["real_days"]])
        temp[dataco_cols["scheduled_days"]] = coerce_numeric(temp[dataco_cols["scheduled_days"]])
        temp["DelayDays"] = temp[dataco_cols["real_days"]] - temp[dataco_cols["scheduled_days"]]
        delay_by_mode = temp.groupby(dataco_cols["shipping_mode"], as_index=False)["DelayDays"].mean().sort_values("DelayDays", ascending=False)
        fig = px.bar(delay_by_mode, x=dataco_cols["shipping_mode"], y="DelayDays", title="Shipment Delays by Shipping Mode")
        show_chart(fig, "Shipment Delays", "Delay by shipping mode helps logistics teams renegotiate carrier policies and improve promised-date accuracy.")
else:
    show_note("Supply Chain Intelligence", "DataCo dataset is unavailable, so operational KPI visualizations were skipped.")
'@

Add-Markdown @'
## 15. LLM Integration

This section integrates a free-tier compatible LLM workflow using `google-generativeai` and Gemini.

Security note: the notebook does not hard-code API keys. Set your key as an environment variable:

```bash
GEMINI_API_KEY=your_api_key_here
```

The LLM receives summarized model, forecast, anomaly, and KPI outputs rather than raw data. This is closer to a production-safe pattern because it limits token usage and avoids unnecessary exposure of sensitive operational records.
'@

Add-Code @'
def build_llm_summary():
    forecast_summary = {
        f"{horizon}_day_forecast_demand": float(future_forecast_df.head(horizon)["ForecastDemand"].sum())
        for horizon in FORECAST_HORIZONS
    }
    anomaly_summary = {
        "demand_anomaly_count": int(len(demand_anomalies)),
        "demand_spikes": int((demand_anomalies["AnomalyType"] == "Demand Spike").sum()) if len(demand_anomalies) else 0,
        "demand_drops": int((demand_anomalies["AnomalyType"] == "Demand Drop").sum()) if len(demand_anomalies) else 0,
        "inventory_anomaly_count": int(len(inventory_anomalies)) if isinstance(inventory_anomalies, pd.DataFrame) else 0,
    }
    summary = {
        "project": PROJECT_NAME,
        "forecasting_source": forecasting_source,
        "best_model": best_model_name,
        "best_model_metrics": {
            "MAE": float(best_row["MAE"]),
            "RMSE": float(best_row["RMSE"]),
            "MAPE": float(best_row["MAPE"]),
            "R2": float(best_row["R2"]),
        },
        "model_ranking": metrics_df.to_dict(orient="records"),
        "forecast_summary": forecast_summary,
        "anomaly_summary": anomaly_summary,
        "supply_chain_kpis": dataco_kpis,
    }
    return summary


llm_summary = build_llm_summary()

consultant_prompt = f"""
You are a Senior Supply Chain Consultant.

Analyze the forecasting output, anomaly detection results, inventory metrics, and shipment metrics.

Provide:

1. Executive Summary
2. Key Insights
3. Risks
4. Inventory Recommendations
5. Procurement Recommendations
6. Logistics Recommendations
7. Cost Optimization Suggestions
8. Future Business Strategy

Use clear business language suitable for executives.

Data summary:
{json.dumps(llm_summary, indent=2, default=str)}
"""

display(Markdown("### LLM Prompt Preview"))
display(Markdown(f"```text\n{consultant_prompt[:4000]}\n```"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        gemini_response = gemini_model.generate_content(consultant_prompt)
        llm_response_text = gemini_response.text
        display(Markdown("### Gemini Executive Recommendation"))
        display(Markdown(llm_response_text))
    except Exception as exc:
        llm_response_text = f"Gemini call failed: {exc}"
        display(Markdown(f"**Gemini call failed:** {exc}"))
else:
    llm_response_text = "Gemini API key not configured. Set GEMINI_API_KEY to enable live LLM recommendations."
    display(Markdown("""
### Gemini Executive Recommendation

Gemini API key not configured, so the live LLM call was skipped.

To enable it:

```python
import os
os.environ["GEMINI_API_KEY"] = "your_api_key_here"
```

Then rerun this cell.
"""))
'@

Add-Markdown @'
## 16. Executive Dashboard

The dashboard condenses the notebook into executive KPIs:

- Forecast demand
- Best model
- RMSE
- Anomaly count
- Inventory risk
- Shipment risk
'@

Add-Code @'
forecast_30d = future_forecast_df.head(30)["ForecastDemand"].sum()
forecast_60d = future_forecast_df.head(60)["ForecastDemand"].sum()
forecast_90d = future_forecast_df.head(90)["ForecastDemand"].sum()
anomaly_count = len(demand_anomalies)
inventory_risk = len(inventory_anomalies) / max(len(dataco_df), 1) if dataco_df is not None and isinstance(inventory_anomalies, pd.DataFrame) else np.nan
shipment_risk = dataco_kpis.get("Late Delivery Rate", np.nan)

cards = [
    ("30-Day Forecast Demand", f"{forecast_30d:,.0f}", "Expected near-term replenishment demand"),
    ("60-Day Forecast Demand", f"{forecast_60d:,.0f}", "Medium-term procurement visibility"),
    ("90-Day Forecast Demand", f"{forecast_90d:,.0f}", "Quarterly capacity planning signal"),
    ("Best Model", best_model_name, "Lowest RMSE on holdout period"),
    ("RMSE", f"{best_row['RMSE']:,.2f}", "Large-error-sensitive accuracy metric"),
    ("Anomaly Count", f"{anomaly_count:,}", "Demand exceptions requiring review"),
    ("Inventory Risk", "N/A" if pd.isna(inventory_risk) else f"{inventory_risk:.2%}", "Proxy based on order/inventory anomalies"),
    ("Shipment Risk", "N/A" if pd.isna(shipment_risk) else f"{shipment_risk:.2%}", "Late delivery exposure"),
]

html_cards = """
<style>
.kpi-grid {display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 14px; margin: 12px 0 24px 0;}
.kpi-card {background: linear-gradient(135deg, #0f172a, #164e63); color: white; padding: 18px; border-radius: 16px; box-shadow: 0 8px 22px rgba(15, 23, 42, 0.18);}
.kpi-title {font-size: 13px; opacity: 0.82; text-transform: uppercase; letter-spacing: .06em;}
.kpi-value {font-size: 26px; font-weight: 800; margin-top: 8px;}
.kpi-note {font-size: 12px; opacity: 0.75; margin-top: 8px;}
</style>
<div class="kpi-grid">
"""
for title, value, note in cards:
    html_cards += f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """
html_cards += "</div>"
display(HTML(html_cards))

fig = make_subplots(
    rows=2,
    cols=3,
    specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
           [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
    subplot_titles=["30-Day Forecast", "60-Day Forecast", "90-Day Forecast", "RMSE", "Anomaly Count", "Shipment Risk"],
)
fig.add_trace(go.Indicator(mode="number", value=forecast_30d, number={"valueformat": ",.0f"}), row=1, col=1)
fig.add_trace(go.Indicator(mode="number", value=forecast_60d, number={"valueformat": ",.0f"}), row=1, col=2)
fig.add_trace(go.Indicator(mode="number", value=forecast_90d, number={"valueformat": ",.0f"}), row=1, col=3)
fig.add_trace(go.Indicator(mode="number", value=float(best_row["RMSE"]), number={"valueformat": ",.2f"}), row=2, col=1)
fig.add_trace(go.Indicator(mode="number", value=anomaly_count, number={"valueformat": ",.0f"}), row=2, col=2)
fig.add_trace(go.Indicator(mode="number", value=0 if pd.isna(shipment_risk) else shipment_risk * 100, number={"suffix": "%", "valueformat": ".2f"}), row=2, col=3)
fig.update_layout(title="Executive KPI Dashboard", template=PLOTLY_TEMPLATE, height=620)
fig.show()
'@

Add-Markdown @'
## 17. Business Impact

### Retail

Retailers can use demand forecasts to improve replenishment, reduce stockouts, optimize promotional inventory, and better allocate stock across stores or regions.

### E-commerce

E-commerce teams can forecast order volume, identify abnormal demand spikes, improve fulfillment planning, and reduce late-delivery risk through proactive logistics signals.

### Manufacturing

Manufacturers can align production plans with predicted demand, reduce raw material shortages, coordinate supplier lead times, and minimize excess finished-goods inventory.

### Logistics

Logistics teams can use shipment KPIs and delay anomalies to identify carrier risk, improve promised-date accuracy, reduce expedite costs, and prioritize high-risk lanes.

### Supply Chain Leadership

Executives gain a single intelligence layer connecting forecasts, operational KPIs, anomalies, and LLM-generated recommendations. This reduces manual analysis time and supports faster, evidence-based decisions.
'@

Add-Markdown @'
## 18. Future Roadmap

### Product Evolution Phases

| Phase | Capability | Outcome |
|---:|---|---|
| Phase 1 | Notebook Prototype | Research-grade proof of concept |
| Phase 2 | Python Application | Modular package with reusable pipelines |
| Phase 3 | Dashboard | Interactive executive and planner views |
| Phase 4 | SaaS Platform | Multi-tenant deployment and role-based access |
| Phase 5 | OMS Integration | Order ingestion and order-risk intelligence |
| Phase 6 | WMS Integration | Inventory, picking, packing, and warehouse signals |
| Phase 7 | TMS Integration | Shipment, carrier, route, and delay intelligence |
| Phase 8 | RAG System | Retrieval over policies, SOPs, contracts, and supplier documents |
| Phase 9 | Agentic AI | Autonomous monitoring, alerts, and workflow triggers |
| Phase 10 | Multi-Agent Supply Chain Copilot | Specialized agents for demand, inventory, procurement, logistics, and finance |

### Target Architecture

```text
OMS + WMS + TMS + ERP + Inventory Systems + Supplier Systems
        |
        v
API Gateway + Streaming Ingestion
        |
        v
Data Lake / Lakehouse
        |
        v
Feature Store + Data Quality Layer
        |
        v
Forecast Engine
        |
        v
Anomaly Engine
        |
        v
LLM / RAG / Agentic AI Layer
        |
        v
Executive Dashboard + Planner Workbench + AI Copilot
```

### Mermaid Architecture Diagram

```mermaid
flowchart TD
    A[OMS] --> F[API Gateway / Streams]
    B[WMS] --> F
    C[TMS] --> F
    D[ERP] --> F
    E[Supplier Systems] --> F
    F --> G[Data Lake / Lakehouse]
    G --> H[Feature Store]
    H --> I[Forecast Engine]
    H --> J[Anomaly Engine]
    I --> K[LLM Layer]
    J --> K
    K --> L[Dashboard]
    K --> M[AI Copilot]
```

### Future Research Extensions

- Hierarchical forecasting by SKU, category, store, region, and market.
- Probabilistic forecasts with prediction intervals for safety-stock planning.
- Causal promotion and price elasticity modeling.
- Supplier risk scoring using external signals.
- Real-time anomaly detection with streaming pipelines.
- RAG-based supply chain policy assistant.
- Multi-agent orchestration for procurement, logistics, and inventory workflows.
'@

Add-Markdown @'
## Executive Conclusion

This notebook demonstrates a complete AI-powered supply chain intelligence prototype. It connects data ingestion, quality profiling, exploratory analytics, feature engineering, statistical forecasting, machine learning forecasting, deep learning forecasting, model benchmarking, future demand forecasting, anomaly detection, supply chain KPIs, LLM-assisted recommendations, and a future SaaS roadmap.

The next engineering step is to refactor the notebook into modular services:

- `data_ingestion`
- `feature_engineering`
- `forecasting`
- `anomaly_detection`
- `kpi_engine`
- `llm_recommendations`
- `dashboard_api`

That transition would move the project from a review-ready research prototype toward an enterprise-grade supply chain intelligence platform.
'@

$notebook = [ordered]@{
    cells = $cells
    metadata = [ordered]@{
        kernelspec = [ordered]@{
            display_name = "Python 3"
            language = "python"
            name = "python3"
        }
        language_info = [ordered]@{
            name = "python"
            version = "3.11"
            mimetype = "text/x-python"
            codemirror_mode = [ordered]@{
                name = "ipython"
                version = 3
            }
            pygments_lexer = "ipython3"
            nbconvert_exporter = "python"
            file_extension = ".py"
        }
    }
    nbformat = 4
    nbformat_minor = 5
}

$outputPath = Join-Path (Get-Location) "AI_Powered_Supply_Chain_Intelligence_Platform.ipynb"
$json = $notebook | ConvertTo-Json -Depth 100
Set-Content -LiteralPath $outputPath -Value $json -Encoding UTF8
Write-Host "Created notebook: $outputPath"
Write-Host "Cell count: $($cells.Count)"
