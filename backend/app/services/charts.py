import pandas as pd

from app.models.schemas import ChartData, ChartSeries


COLORS = [
    "#2563eb",
    "#0891b2",
    "#059669",
    "#d97706",
    "#dc2626",
    "#7c3aed",
    "#db2777",
]


def _series(label: str, data: list, color_idx: int = 0, fill: bool = False) -> ChartSeries:
    return ChartSeries(
        label=label,
        data=[float(x) if pd.notna(x) else None for x in data],
        borderColor=COLORS[color_idx % len(COLORS)],
        backgroundColor=COLORS[color_idx % len(COLORS)] + "33" if fill else COLORS[color_idx % len(COLORS)],
        fill=fill,
    )


def build_charts(
    forecasting_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    prediction_frames: dict[str, pd.DataFrame],
    future_forecast_df: pd.DataFrame,
    demand_anomalies: pd.DataFrame,
    dataco_df: pd.DataFrame | None,
    kpis: dict,
    best_model_name: str,
) -> list[ChartData]:
    charts: list[ChartData] = []
    ts = forecasting_df.copy()
    ts["Date"] = pd.to_datetime(ts["Date"])
    ts = ts.sort_values("Date")

    # 1. Demand trend
    charts.append(
        ChartData(
            id="demand_trend",
            title="Demand Trend",
            type="line",
            labels=[d.strftime("%Y-%m-%d") for d in ts["Date"]],
            datasets=[_series("Demand", ts["Demand"].tolist())],
            interpretation="Long-term direction for replenishment and capacity planning.",
        )
    )

    # 2. Monthly demand
    monthly = ts.set_index("Date")["Demand"].resample("ME").sum().reset_index()
    charts.append(
        ChartData(
            id="monthly_demand",
            title="Monthly Demand",
            type="bar",
            labels=[d.strftime("%Y-%m") for d in monthly["Date"]],
            datasets=[_series("Monthly Demand", monthly["Demand"].tolist(), fill=True)],
            interpretation="Monthly peaks support S&OP and procurement cycles.",
        )
    )

    # 3. Weekly demand
    weekly = ts.set_index("Date")["Demand"].resample("W").sum().reset_index()
    charts.append(
        ChartData(
            id="weekly_demand",
            title="Weekly Demand",
            type="line",
            labels=[d.strftime("%Y-%m-%d") for d in weekly["Date"]],
            datasets=[_series("Weekly Demand", weekly["Demand"].tolist(), color_idx=1)],
            interpretation="Weekly patterns align warehouse labor and transport capacity.",
        )
    )

    # 4. Demand histogram
    charts.append(
        ChartData(
            id="demand_histogram",
            title="Demand Distribution",
            type="bar",
            labels=[f"Bin {i+1}" for i in range(min(20, len(ts)))],
            datasets=[_series("Frequency", _histogram_bins(ts["Demand"], 20), fill=True)],
            interpretation="Distribution shape indicates volatility and safety-stock needs.",
        )
    )

    # 5. Model RMSE comparison
    charts.append(
        ChartData(
            id="model_rmse",
            title="Model RMSE Comparison",
            type="bar",
            labels=metrics_df["Model"].tolist(),
            datasets=[_series("RMSE", metrics_df["RMSE"].tolist(), color_idx=2)],
            interpretation="Lower RMSE means fewer costly large forecast errors.",
        )
    )

    # 6. Best model actual vs predicted
    if best_model_name in prediction_frames:
        pred = prediction_frames[best_model_name]
        charts.append(
            ChartData(
                id="actual_vs_predicted",
                title=f"{best_model_name}: Actual vs Predicted",
                type="line",
                labels=[d.strftime("%Y-%m-%d") for d in pd.to_datetime(pred["Date"])],
                datasets=[
                    _series("Actual", pred["Actual"].tolist()),
                    _series("Predicted", pred["Predicted"].tolist(), color_idx=1),
                ],
                interpretation="Tracks how well the selected model follows real demand.",
            )
        )

    # 7. Future forecast
    hist_window = ts.tail(min(180, len(ts)))
    future = future_forecast_df.copy()
    future["Date"] = pd.to_datetime(future["Date"])
    combined_labels = [d.strftime("%Y-%m-%d") for d in hist_window["Date"]] + [
        d.strftime("%Y-%m-%d") for d in future["Date"]
    ]
    charts.append(
        ChartData(
            id="future_forecast",
            title="Historical vs 90-Day Forecast",
            type="line",
            labels=combined_labels,
            datasets=[
                ChartSeries(
                    label="Historical",
                    data=[float(x) for x in hist_window["Demand"].tolist()] + [None] * len(future),
                    borderColor=COLORS[0],
                ),
                ChartSeries(
                    label="Forecast",
                    data=[None] * len(hist_window) + [float(x) for x in future["ForecastDemand"].tolist()],
                    borderColor=COLORS[4],
                ),
            ],
            interpretation="Forward demand signal for procurement and inventory decisions.",
        )
    )

    # 8. Anomaly scatter overlay as bar counts
    if not demand_anomalies.empty:
        counts = demand_anomalies["AnomalyType"].value_counts()
        charts.append(
            ChartData(
                id="anomaly_counts",
                title="Demand Anomalies",
                type="doughnut",
                labels=counts.index.tolist(),
                datasets=[
                    ChartSeries(
                        label="Count",
                        data=[int(x) for x in counts.values.tolist()],
                        backgroundColor=COLORS[: len(counts)],
                    )
                ],
                interpretation="Spikes vs drops guide stockout and overstock risk response.",
            )
        )

    # 9. KPI radar (normalized)
    kpi_keys = ["Late Delivery Rate", "Profit Margin", "Shipment Delay Rate", "Negative Profit Rate"]
    kpi_vals = []
    kpi_labels = []
    for k in kpi_keys:
        v = kpis.get(k)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            kpi_labels.append(k)
            kpi_vals.append(float(v) * 100 if v <= 1 else float(v))
    if kpi_labels:
        charts.append(
            ChartData(
                id="kpi_radar",
                title="Supply Chain Risk KPIs",
                type="radar",
                labels=kpi_labels,
                datasets=[_series("Risk Exposure", kpi_vals, fill=True)],
                interpretation="Operational risk profile from uploaded supply chain data.",
            )
        )

    # 10. Region profit if DataCo-like
    if dataco_df is not None:
        from app.services.data_loader import find_column, coerce_numeric

        region_col = find_column(dataco_df, preferred=["Order Region", "Customer Region"], contains_any=["region"])
        profit_col = find_column(dataco_df, preferred=["Benefit per order", "Order Profit Per Order"], contains_any=["profit", "benefit"])
        if region_col and profit_col:
            temp = dataco_df[[region_col, profit_col]].copy()
            temp[profit_col] = coerce_numeric(temp[profit_col])
            region_profit = temp.groupby(region_col, as_index=False)[profit_col].sum().sort_values(profit_col, ascending=False).head(10)
            charts.append(
                ChartData(
                    id="profit_by_region",
                    title="Profit by Region",
                    type="bar",
                    labels=region_profit[region_col].astype(str).tolist(),
                    datasets=[_series("Profit", region_profit[profit_col].tolist(), color_idx=3, fill=True)],
                    interpretation="Regional profitability drives localized fulfillment strategy.",
                )
            )

    return charts


def _histogram_bins(series: pd.Series, bins: int) -> list[float]:
    counts, _ = pd.cut(series.dropna(), bins=bins, retbins=True)
    return counts.value_counts(sort=False).tolist()
