import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(layout="wide", page_title="Prévision IA Ventes & Stock")

st.markdown("""
    <style>
    body { background-color: #111; color: #fcd000; }
    .stApp { background-color: #111; }
    .big-font {font-size: 22px !important; font-weight: bold; color: #fcd000;}
    .centered {text-align: center;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Prévision IA des Ventes & Gestion de Stock")

DATA_FILE = "Rocamora Files_SampleData_26062025 (1).xlsx"


def normalize_cai(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=object)
    numeric_series = pd.to_numeric(series, errors="coerce")
    return numeric_series.apply(lambda x: f"{int(x)}" if pd.notna(x) else np.nan)


@st.cache_data
def load_data():
    xls = pd.ExcelFile(DATA_FILE)
    return xls.parse("Daily Sales 2024"), xls.parse("Daily Sales 26062025"), xls.parse("Daily Stock 26062025")

@st.cache_data
def prepare_data(sales_2024, sales_2025, stock_df):
    for df in [sales_2024, sales_2025]:
        df["Billing Date"] = pd.to_datetime(df["Billing Date"], errors="coerce")
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df["CAI"] = normalize_cai(df["CAI"])
    sales = pd.concat([sales_2024, sales_2025], ignore_index=True)
    sales.dropna(subset=["Billing Date", "Quantity", "CAI"], inplace=True)
    sales = sales[(sales["Billing Date"].dt.year >= 2024) & (sales["Billing Date"].dt.year <= 2026)]
    sales["Month"] = sales["Billing Date"].dt.to_period("M")
    sales["Date"] = sales["Month"].dt.to_timestamp()
    sales["Month_Num"] = sales["Date"].dt.month
    sales["Year"] = sales["Date"].dt.year
    sales["Month_Index"] = (sales["Year"] - sales["Year"].min()) * 12 + sales["Month_Num"]
    sales["Month_sin"] = np.sin(2 * np.pi * sales["Month_Num"] / 12)
    sales["Month_cos"] = np.cos(2 * np.pi * sales["Month_Num"] / 12)
    monthly_sales = sales.groupby(["CAI", "Item Description", "Date", "Month_Index", "Month_Num", "Year", "Month_sin", "Month_cos"]).agg(Quantity=("Quantity", "sum")).reset_index()
    stock_cai_series = stock_df["CAI_CODE"] if "CAI_CODE" in stock_df else stock_df.get("CAI")
    stock_df["CAI"] = normalize_cai(stock_cai_series)
    stock_df["QTY"] = pd.to_numeric(stock_df["QTY"], errors="coerce")
    stock_df = stock_df.dropna(subset=["CAI", "QTY"])
    stock_summary = stock_df.groupby(["CAI", "Item Description"]).agg(Stock_QTY=("QTY", "sum")).reset_index()
    return sales, monthly_sales, stock_summary

@st.cache_data
def generate_forecasts(monthly_sales):
    ranking = monthly_sales.groupby("CAI")["Quantity"].sum().sort_values(ascending=False)
    forecast_all = []
    top_items = []
    for item in ranking.index:
        df = monthly_sales[monthly_sales["CAI"] == item].copy().sort_values("Date")
        df["RollingMean_3"] = df["Quantity"].rolling(window=3, min_periods=1).mean()
        features = ["Month_Index", "Month_Num", "Year", "Month_sin", "Month_cos", "RollingMean_3"]
        df = df.dropna(subset=features)
        if df.empty:
            continue
        model = xgb.XGBRegressor(n_estimators=20, max_depth=2, random_state=42, n_jobs=-1)
        model.fit(df[features], df["Quantity"])
        last_date, last_index = df["Date"].max(), df["Month_Index"].max()
        rolling_window = df["Quantity"].values[-3:].tolist()
        for i in range(1, 13):
            future_date = last_date + pd.DateOffset(months=i)
            m, y = future_date.month, future_date.year
            idx = last_index + i
            sin, cos = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
            roll = np.mean(rolling_window[-3:])
            X = pd.DataFrame.from_records([{ "Month_Index": idx, "Month_Num": m, "Year": y, "Month_sin": sin, "Month_cos": cos, "RollingMean_3": roll }])
            pred = model.predict(X)[0]
            rolling_window.append(pred)
            forecast_all.append({
                "CAI": item,
                "Item Description": df["Item Description"].iloc[0],
                "Date": future_date,
                "Predicted Quantity": np.round(pred),
                "IC_lower": max(0, np.round(pred * 0.85)),
                "IC_upper": np.round(pred * 1.15)
            })
        top_items.append(item)
        if len(top_items) >= 10:
            break
    forecast_df = pd.DataFrame(forecast_all)
    return forecast_df, top_items

sales_2024, sales_2025, stock_df = load_data()
sales, monthly_sales, stock_summary = prepare_data(sales_2024, sales_2025, stock_df)
forecast_df, top_items = generate_forecasts(monthly_sales)

item_labels = monthly_sales.groupby("CAI")["Item Description"].first().to_dict()

if not top_items:
    st.warning("Aucune prévision disponible pour les CAI.")
    st.stop()

selected_item = st.selectbox(
    "📦 Sélectionner un CAI du TOP 10 :",
    top_items,
    format_func=lambda cai: f"{cai} – {item_labels.get(cai, '')}".rstrip(" – ")
)
hist_data = monthly_sales[monthly_sales["CAI"] == selected_item].copy()
forecast_data = forecast_df[forecast_df["CAI"] == selected_item].copy()
item_description = item_labels.get(selected_item, "")

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_data["Date"], y=hist_data["Quantity"], mode='lines+markers', name='Historique', line=dict(color='gold')))
fig.add_trace(go.Scatter(x=forecast_data["Date"], y=forecast_data["Predicted Quantity"], mode='lines+markers', name='Prévision IA', line=dict(color='lime')))
fig.add_trace(go.Scatter(x=forecast_data["Date"], y=forecast_data["IC_upper"], name="IC upper", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=forecast_data["Date"], y=forecast_data["IC_lower"], name="IC lower", fill='tonexty', fillcolor='rgba(0,255,0,0.2)', line=dict(width=0), showlegend=False))
title_suffix = f" – {item_description}" if item_description else ""
fig.update_layout(title=f"Prévision de ventes – CAI {selected_item}{title_suffix}", xaxis_title='Mois', yaxis_title='Quantité', template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 Détail du mois de juillet")
july_stock = stock_summary[stock_summary["CAI"] == selected_item]["Stock_QTY"].values
july_stock = july_stock[0] if len(july_stock) > 0 else 0
forecast_item = forecast_data.copy()
forecast_item["Cumul prévisions"] = forecast_item["Predicted Quantity"].cumsum()
mois_rup = forecast_item[forecast_item["Cumul prévisions"] > july_stock]
rupture_text = "✅ Stock suffisant sur les 12 mois." if mois_rup.empty else f"⚠️ Réapprovisionnement nécessaire avant {mois_rup.iloc[0]['Date'].strftime('%B %Y')}"

col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(forecast_item[["Date", "Predicted Quantity", "Cumul prévisions"]].rename(columns={
        "Date": "Mois",
        "Predicted Quantity": "Prévision",
        "Cumul prévisions": "Prévision cumulée"
    }))
with col2:
    st.metric("📦 Stock disponible (juillet)", int(july_stock))
    st.markdown(rupture_text)
