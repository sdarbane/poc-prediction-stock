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

@st.cache_data
def load_data():
    xls = pd.ExcelFile("Rocamora Files_SampleData_26062025.xlsx")
    return xls.parse("Daily Sales 2024"), xls.parse("Daily Sales 26062025"), xls.parse("Daily Stock 26062025")

@st.cache_data
def prepare_data(sales_2024, sales_2025, stock_df):
    for df in [sales_2024, sales_2025]:
        df["Billing Date"] = pd.to_datetime(df["Billing Date"], errors="coerce")
    sales = pd.concat([sales_2024, sales_2025], ignore_index=True)
    sales.dropna(subset=["Billing Date", "Quantity", "Item Code"], inplace=True)
    sales = sales[(sales["Billing Date"].dt.year >= 2024) & (sales["Billing Date"].dt.year <= 2026)]
    sales["Month"] = sales["Billing Date"].dt.to_period("M")
    sales["Date"] = sales["Month"].dt.to_timestamp()
    sales["Month_Num"] = sales["Date"].dt.month
    sales["Year"] = sales["Date"].dt.year
    sales["Month_Index"] = (sales["Year"] - sales["Year"].min()) * 12 + sales["Month_Num"]
    sales["Month_sin"] = np.sin(2 * np.pi * sales["Month_Num"] / 12)
    sales["Month_cos"] = np.cos(2 * np.pi * sales["Month_Num"] / 12)
    monthly_sales = sales.groupby(["Item Code", "Item Description", "Date", "Month_Index", "Month_Num", "Year", "Month_sin", "Month_cos"]).agg(Quantity=("Quantity", "sum")).reset_index()
    stock_df["QTY"] = pd.to_numeric(stock_df["QTY"], errors="coerce")
    stock_summary = stock_df.groupby(["ITEM_CODE", "Item Description"]).agg(Stock_QTY=("QTY", "sum")).reset_index()
    return sales, monthly_sales, stock_summary

@st.cache_data
def generate_forecasts(monthly_sales):
    top_items = monthly_sales.groupby("Item Code")["Quantity"].sum().sort_values(ascending=False).head(10).index.tolist()
    forecast_all = []
    for item in top_items:
        df = monthly_sales[monthly_sales["Item Code"] == item].copy().sort_values("Date")
        df["RollingMean_3"] = df["Quantity"].rolling(window=3, min_periods=1).mean()
        features = ["Month_Index", "Month_Num", "Year", "Month_sin", "Month_cos", "RollingMean_3"]
        df = df.dropna(subset=features)
        model = xgb.XGBRegressor(n_estimators=30, max_depth=3, random_state=42, n_jobs=-1)
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
                "Item Code": item,
                "Item Description": df["Item Description"].iloc[0],
                "Date": future_date,
                "Predicted Quantity": pred,
                "IC_80_lower": max(0, pred * 0.90),
                "IC_80_upper": pred * 1.10,
                "IC_95_lower": max(0, pred * 0.85),
                "IC_95_upper": pred * 1.15
            })
    forecast_df = pd.DataFrame(forecast_all)
    return forecast_df, top_items

sales_2024, sales_2025, stock_df = load_data()
sales, monthly_sales, stock_summary = prepare_data(sales_2024, sales_2025, stock_df)
forecast_df, top_items = generate_forecasts(monthly_sales)

selected_item = st.selectbox("📦 Sélectionner un produit du TOP 10 :", top_items)
hist_data = monthly_sales[monthly_sales["Item Code"] == selected_item].copy()
forecast_data = forecast_df[forecast_df["Item Code"] == selected_item].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_data["Date"], y=hist_data["Quantity"], mode='lines+markers', name='Historique', line=dict(color='gold')))
fig.add_trace(go.Scatter(x=forecast_data["Date"], y=forecast_data["Predicted Quantity"], mode='lines+markers', name='Prévision IA', line=dict(color='lime')))
fig.add_trace(go.Scatter(x=forecast_data["Date"], y=forecast_data["IC_95_upper"], name="IC 95%+", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=forecast_data["Date"], y=forecast_data["IC_95_lower"], name="IC 95%-", fill='tonexty', fillcolor='rgba(0,255,0,0.1)', line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=forecast_data["Date"], y=forecast_data["IC_80_upper"], name="IC 80%+", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=forecast_data["Date"], y=forecast_data["IC_80_lower"], name="IC 80%-", fill='tonexty', fillcolor='rgba(0,255,0,0.2)', line=dict(width=0), showlegend=False))
fig.update_layout(title=f"Prévision de ventes – {selected_item}", xaxis_title='Mois', yaxis_title='Quantité', template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 Détail du mois de juillet")
july_stock = stock_summary[stock_summary["ITEM_CODE"] == selected_item]["Stock_QTY"].values
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
