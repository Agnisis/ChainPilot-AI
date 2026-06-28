import re
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def find_column(
    df: pd.DataFrame,
    preferred: list[str] | None = None,
    contains_any: list[str] | None = None,
    contains_all: list[str] | None = None,
) -> str | None:
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


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def read_csv_resilient(path: Path, nrows: int | None = None) -> pd.DataFrame:
    encodings = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, nrows=nrows, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error  # type: ignore[misc]


def prepare_m5_timeseries(calendar: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame | None:
    if calendar is None or sales is None:
        return None
    d_cols = sorted(
        [c for c in sales.columns if re.fullmatch(r"d_\d+", str(c))],
        key=lambda x: int(x.split("_")[1]),
    )
    if not d_cols:
        return None
    demand_by_d = (
        sales[d_cols].sum(axis=0).rename("Demand").reset_index().rename(columns={"index": "d"})
    )
    calendar_cols = [
        c
        for c in ["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year"]
        if c in calendar.columns
    ]
    calendar_small = calendar[calendar_cols].copy()
    calendar_small["date"] = pd.to_datetime(calendar_small["date"], errors="coerce")
    ts = calendar_small.merge(demand_by_d, on="d", how="inner")
    ts = ts.rename(columns={"date": "Date"})
    return ts.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def detect_date_target_columns(
    df: pd.DataFrame, date_override: str | None = None, target_override: str | None = None
) -> tuple[str | None, str | None]:
    date_col = date_override or find_column(
        df,
        preferred=["date", "order date", "order date (DateOrders)", "shipping date (DateOrders)"],
        contains_any=["dateorders", "order date", "date"],
    )
    target_col = target_override or find_column(
        df,
        preferred=["Sales", "Order Item Quantity", "Order Item Total", "Demand", "Quantity"],
        contains_any=["sales", "quantity", "demand"],
    )
    return date_col, target_col


def prepare_generic_timeseries(
    df: pd.DataFrame, date_col: str | None = None, target_col: str | None = None
) -> pd.DataFrame | None:
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
        temp.groupby(pd.Grouper(key=date_col, freq="D"))[target_col]
        .sum()
        .reset_index()
        .rename(columns={date_col: "Date", target_col: "Demand"})
    )
    daily = daily[daily["Demand"].notna()].sort_values("Date").reset_index(drop=True)
    # Cap at last 365 days to ensure blazing fast CPU execution for the demo
    if len(daily) > 365:
        daily = daily.tail(365).reset_index(drop=True)
    return daily


def prepare_rossmann_timeseries(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame | None:
    if train is None: return None
    temp = train[["Date", "Sales"]].copy()
    temp["Date"] = pd.to_datetime(temp["Date"], errors="coerce")
    temp["Sales"] = pd.to_numeric(temp["Sales"], errors="coerce")
    temp = temp.dropna(subset=["Date", "Sales"])
    daily = temp.groupby(pd.Grouper(key="Date", freq="D"))["Sales"].sum().reset_index()
    daily = daily.rename(columns={"Sales": "Demand"})
    return daily.sort_values("Date").reset_index(drop=True)

def prepare_olist_timeseries(orders: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame | None:
    if orders is None or items is None: return None
    merged = pd.merge(items, orders, on="order_id", how="inner")
    merged["order_purchase_timestamp"] = pd.to_datetime(merged["order_purchase_timestamp"], errors="coerce")
    merged["price"] = pd.to_numeric(merged["price"], errors="coerce")
    merged = merged.dropna(subset=["order_purchase_timestamp", "price"])
    daily = merged.groupby(pd.Grouper(key="order_purchase_timestamp", freq="D"))["price"].sum().reset_index()
    daily = daily.rename(columns={"order_purchase_timestamp": "Date", "price": "Demand"})
    return daily.sort_values("Date").reset_index(drop=True)

def load_session_datasets(session_dir: Path) -> dict[str, pd.DataFrame | None]:
    files = list(session_dir.glob("*.csv"))
    result: dict[str, pd.DataFrame | None] = {
        "calendar": None,
        "sales": None,
        "sell_prices": None,
        "dataco": None,
        "rossmann_train": None,
        "rossmann_store": None,
        "olist_orders": None,
        "olist_items": None,
        "generic": None,
    }
    for path in files:
        name = path.name.lower()
        df = read_csv_resilient(path)
        if "calendar" in name:
            result["calendar"] = df
        elif "sales_train" in name or (name.startswith("sales") and not "store" in name):
            result["sales"] = df
        elif "sell_price" in name:
            result["sell_prices"] = df
        elif "dataco" in name or "supplychain" in name:
            result["dataco"] = df
        elif name == "train.csv":
            result["rossmann_train"] = df
        elif name == "store.csv":
            result["rossmann_store"] = df
        elif "olist_orders_dataset" in name:
            result["olist_orders"] = df
        elif "olist_order_items_dataset" in name:
            result["olist_items"] = df
        else:
            result["generic"] = df
    return result


def build_forecasting_frame(
    datasets: dict[str, pd.DataFrame | None],
    date_column: str | None = None,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, str, pd.DataFrame | None]:
    if datasets.get("rossmann_train") is not None:
        ts = prepare_rossmann_timeseries(datasets.get("rossmann_train"), datasets.get("rossmann_store"))
        if ts is not None and len(ts) >= 30:
            if len(ts) > 365: ts = ts.tail(365).reset_index(drop=True)
            return ts, "Rossmann Retail Sales", datasets.get("rossmann_train")
            
    if datasets.get("olist_orders") is not None:
        ts = prepare_olist_timeseries(datasets.get("olist_orders"), datasets.get("olist_items"))
        if ts is not None and len(ts) >= 30:
            if len(ts) > 365: ts = ts.tail(365).reset_index(drop=True)
            return ts, "Olist E-Commerce Demand", datasets.get("olist_orders")

    m5_ts = prepare_m5_timeseries(datasets.get("calendar"), datasets.get("sales"))
    dataco_df = datasets.get("dataco")
    if dataco_df is None:
        dataco_df = datasets.get("generic")
    dataco_ts = prepare_generic_timeseries(dataco_df, date_column, target_column)

    if m5_ts is not None and len(m5_ts) >= 180:
        return m5_ts[["Date", "Demand"]].copy(), "M5 aggregated daily demand", dataco_df
    if dataco_ts is not None and len(dataco_ts) >= 180:
        return dataco_ts[["Date", "Demand"]].copy(), "Uploaded dataset aggregated daily demand", dataco_df
    if dataco_ts is not None and len(dataco_ts) >= 30:
        return dataco_ts[["Date", "Demand"]].copy(), "Uploaded dataset (short series)", dataco_df
    raise ValueError(
        "Could not build forecasting series. Upload a CSV with date and demand/sales/quantity columns, "
        "or M5 calendar + sales files."
    )
