import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time

st.set_page_config(page_title="Prévision des ventes", layout="wide")
st.markdown("""
    <style>
        .stApp {background-color: #0f0f0f; color: #ffffff; font-family: 'Segoe UI', sans-serif;}
        .css-1aumxhk {background-color: #1e1e1e; border: none;}
        .css-1d391kg, .css-ffhzg2 {color: #f1c40f;}
    </style>
""", unsafe_allow_html=True)

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
def generate_forecasts(grouped):
    all_predictions = {}
    top_items = grouped.groupby('Item_Code')['Quantity'].sum().sort_values(ascending=False).head(10).index
    for item in top_items:
        item_df = grouped[grouped['Item_Code'] == item].copy()
        item_df = item_df.sort_values('Month')
        item_df['Month_idx'] = np.arange(len(item_df))

        # Features: month idx + rolling mean
        item_df['RollingMean'] = item_df['Quantity'].rolling(window=3, min_periods=1).mean()

        # Model
        X = item_df[['Month_idx', 'RollingMean']]
        y = item_df['Quantity']

        model = XGBRegressor(n_estimators=50)
        model.fit(X, y)

        future_months = [item_df['Month'].max() + relativedelta(months=i+1) for i in range(12)]
        future_idx = np.arange(len(item_df), len(item_df) + 12)
        last_rm = item_df['RollingMean'].iloc[-1]
        future_rm = [last_rm] * 12

        future_X = pd.DataFrame({'Month_idx': future_idx, 'RollingMean': future_rm})
        preds = model.predict(future_X)

        # Confidence Intervals (simple ±5% and ±10%)
        ci_5 = preds * 0.05
        ci_10 = preds * 0.10

        all_predictions[item] = pd.DataFrame({
            'Month': future_months,
            'Prediction': preds,
            'CI_lower_5': preds - ci_5,
            'CI_upper_5': preds + ci_5,
            'CI_lower_10': preds - ci_10,
            'CI_upper_10': preds + ci_10
        })
    return all_predictions, top_items

# Chargement des données
with st.spinner("Chargement et entraînement du modèle IA..."):
    grouped, stock = prepare_data()
    forecasts, top_items = generate_forecasts(grouped)
    time.sleep(1)

# Interface utilisateur
item_selected = st.selectbox("\U0001F4C5 Sélectionner un produit du TOP 10 :", top_items)
preds_df = forecasts[item_selected]
hist_df = grouped[grouped['Item_Code'] == item_selected]

# Affichage du graphique
ci_option = st.radio("Intervalle de confiance :", ["±5%", "±10%"], horizontal=True)
ci_col = "CI_lower_5" if ci_option == "±5%" else "CI_lower_10"
ci_col_u = "CI_upper_5" if ci_option == "±5%" else "CI_upper_10"

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_df['Month'], y=hist_df['Quantity'], name='Historique',
                         mode='lines+markers', line=dict(color='yellow')))
fig.add_trace(go.Scatter(x=preds_df['Month'], y=preds_df['Prediction'],
                         mode='lines+markers', name='Prévision IA',
                         line=dict(color='lime')))
fig.add_trace(go.Scatter(x=preds_df['Month'].tolist() + preds_df['Month'][::-1].tolist(),
                         y=preds_df[ci_col_u].tolist() + preds_df[ci_col][::-1].tolist(),
                         fill='toself', fillcolor='rgba(50,205,50,0.2)',
                         line=dict(color='rgba(255,255,255,0)'), name='Intervalle confiance'))

fig.update_layout(title=f"Prévision de ventes – {item_selected}", xaxis_title="Mois", yaxis_title="Quantité",
                  template='plotly_dark', legend=dict(x=1, y=1))
st.plotly_chart(fig, use_container_width=True)

# Stock & BO
st.subheader("\U0001F4C4 Détail du mois de juillet")
stock_row = stock[stock['Item_Code'] == item_selected]
stock_value = stock_row['Quantity'].values[0] if not stock_row.empty else 0

merged = preds_df.copy()
merged["Prévision"] = merged["Prediction"].round(0).astype(int)
merged = merged[["Month", "Prévision"]].copy()
merged["Stock disponible"] = stock_value
merged["Cumul prévision"] = merged["Prévision"].cumsum()
merged["Risque BO"] = merged["Cumul prévision"] > stock_value
merged["Alerte"] = merged["Risque BO"].apply(lambda x: "❌ Réappro" if x else "✅ OK")

# Affichage tableau coloré
st.dataframe(merged.style.applymap(
    lambda val: "background-color: red; color: white" if val == "❌ Réappro" else "background-color: green; color: white",
    subset=["Alerte"]
), use_container_width=True)

# Footer
st.markdown("""
    <style>
        .css-15zrgzn {color: #999 !important;}
    </style>
""", unsafe_allow_html=True)

st.success("Analyse terminée ✨ Vous pouvez explorer les prévisions par produit.")
