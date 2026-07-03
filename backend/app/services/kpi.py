import numpy as np
import pandas as pd

from app.services.data_loader import coerce_numeric, find_column


def compute_supply_chain_kpis(df: pd.DataFrame | None) -> dict[str, float | int | None]:
    if df is None or df.empty:
        return {}

    cols = {
        "order_id": find_column(df, preferred=["Order Id", "Order ID"], contains_all=["order", "id"]),
        "sales": find_column(df, preferred=["Sales", "Order Item Total"], contains_any=["sales", "total"]),
        "quantity": find_column(df, preferred=["Order Item Quantity", "Quantity"], contains_any=["quantity"]),
        "profit": find_column(df, preferred=["Benefit per order", "Order Profit Per Order"], contains_any=["profit", "benefit"]),
        "late_risk": find_column(df, preferred=["Late_delivery_risk"], contains_all=["late", "risk"]),
        "delivery_status": find_column(df, preferred=["Delivery Status"], contains_all=["delivery", "status"]),
        "real_days": find_column(df, preferred=["Days for shipping (real)"], contains_all=["days", "shipping"]),
        "scheduled_days": find_column(df, preferred=["Days for shipment (scheduled)"], contains_all=["days", "scheduled"]),
    }

    working = df.copy()
    for key in ["sales", "quantity", "profit", "late_risk", "real_days", "scheduled_days"]:
        if cols[key]:
            working[cols[key]] = coerce_numeric(working[cols[key]])

    total_orders = working[cols["order_id"]].nunique() if cols["order_id"] else len(working)
    total_sales = float(working[cols["sales"]].sum()) if cols["sales"] else None
    total_units = float(working[cols["quantity"]].sum()) if cols["quantity"] else None
    total_profit = float(working[cols["profit"]].sum()) if cols["profit"] else None
    avg_order_value = total_sales / total_orders if cols["sales"] and total_orders else None
    profit_margin = total_profit / total_sales if cols["profit"] and cols["sales"] and total_sales else None

    if cols["late_risk"]:
        late_rate = float(working[cols["late_risk"]].mean())
    elif cols["delivery_status"]:
        late_rate = float(working[cols["delivery_status"]].astype(str).str.lower().str.contains("late").mean())
    else:
        late_rate = None

    if cols["real_days"] and cols["scheduled_days"]:
        delay_days = working[cols["real_days"]] - working[cols["scheduled_days"]]
        avg_delay = float(delay_days.mean())
        delay_rate = float((delay_days > 0).mean())
    else:
        avg_delay = None
        delay_rate = None

    negative_profit_rate = float((working[cols["profit"]] < 0).mean()) if cols["profit"] else None

    return {
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


def format_kpi_value(value, pct=False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if pct:
        return f"{value:.2%}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}"
