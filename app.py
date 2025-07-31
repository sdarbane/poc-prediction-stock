import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime
import time

# Configuration de page
st.set_page_config(layout="wide", page_title="Prévision IA Ventes & Stock")
st.markdown("""
    <style>
    body { background-color: #111; color: #fcd000; }
    .stApp { background-color: #111; }
    .big-font {font-size: 22px !important; font-weight: bold; color: #fcd000;}
    .centered {text-align: center;}
    .plotly-graph {background-color: #222; border-radius: 8px; padding: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Prévision IA des Ventes & Gestion de Stock")

@st.cache_data
def load_data():
    xls = pd.ExcelFile("Rocamora Files_SampleData_26062025.xlsx")
    return xls.parse("Daily Sales 2024"), xls.parse("Daily Sales 26062025"), xls.parse("Daily Stock 26062025")

with st.spinner("📥 Chargement des données en cours..."):
    time.sleep(1)
    sales_2024, sales_2025, stock_df = load_data()
st.success("✅ Données chargées avec succès !")

with st.spinner("🧠 Prétraitement et entraînement du modèle en cours..."):
    time.sleep(1.2)

    for df in [sales_2024, sales_2025]:
        df["Billing Date"] = pd.to_datetime(df["Billing Date"], errors="coerce")

    sales = pd.concat([sales_2024, sales_2025], ignore_index=True)
    sales.dropna(subset=["Billing Date", "Quantity", "Item Code"], inplace=True)
    sales["Month"] = sales["Billing Date"].dt.to_period("M")
    sales["Date"] = sales["Month"].dt.to_timestamp()
    sales["Month_Num"] = sales["Date"].dt.month
    sales["Year"] = sales["Date"].dt.year
    sales["Month_Index"] = (sales["Year"] - sales["Year"].min()) * 12 + sales["Month_Num"]
    sales["Month_sin"] = np.sin(2 * np.pi * sales["Month_Num"] / 12)
    sales["Month_cos"] = np.cos(2 * np.pi * sales["Month_Num"] / 12)

    monthly_sales = sales.groupby(["Item Code", "Item Description", "Date", "Month_Index", "Month_Num", "Year", "Month_sin", "Month_cos"]).agg(
        Quantity=("Quantity", "sum")
    ).reset_index()

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
                "IC_5_lower": max(0, pred * 0.95),
                "IC_5_upper": pred * 1.05,
                "IC_10_lower": max(0, pred * 0.90),
                "IC_10_upper": pred * 1.10
            })

forecast_df = pd.DataFrame(forecast_all)
stock_df["QTY"] = pd.to_numeric(stock_df["QTY"], errors="coerce")
stock_summary = stock_df.groupby(["ITEM_CODE", "Item Description"]).agg(Stock_QTY=("QTY", "sum")).reset_index()
st.success("✅ Modèle entraîné et prévisions générées pour le TOP 10 produits.")

# Interface utilisateur
st.markdown("### 🔎 Visualisation des prévisions")
ic_range = st.radio("Choisir un intervalle de confiance :", ["±5%", "±10%"])
selected_index = st.slider("🔟 Choisissez un produit parmi le TOP 10 :", 0, len(top_items)-1, 0)
selected_item = top_items[selected_index]

item_data = forecast_df[forecast_df["Item Code"] == selected_item].copy()
past_data = monthly_sales[monthly_sales["Item Code"] == selected_item][["Date", "Quantity"]].copy()
stock_july = stock_summary[stock_summary["ITEM_CODE"] == selected_item]["Stock_QTY"].values[0]

# Graphique interactif
fig = go.Figure()
fig.add_trace(go.Scatter(x=past_data["Date"], y=past_data["Quantity"], mode="lines+markers", name="Historique", line=dict(color="#fcd000")))
fig.add_trace(go.Scatter(x=item_data["Date"], y=item_data["Predicted Quantity"], mode="lines+markers", name="Prévision IA", line=dict(color="#00cc96")))

if ic_range == "±5%":
    fig.add_trace(go.Scatter(x=item_data["Date"], y=item_data["IC_5_upper"], name="IC +5%", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=item_data["Date"], y=item_data["IC_5_lower"], name="IC -5%", fill='tonexty', line=dict(width=0), fillcolor='rgba(0,204,150,0.2)', showlegend=False))
else:
    fig.add_trace(go.Scatter(x=item_data["Date"], y=item_data["IC_10_upper"], name="IC +10%", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=item_data["Date"], y=item_data["IC_10_lower"], name="IC -10%", fill='tonexty', line=dict(width=0), fillcolor='rgba(0,204,150,0.2)', showlegend=False))

fig.update_layout(title=f"📦 {selected_item} – Prévision des ventes avec intervalle de confiance",
                  xaxis_title="Mois", yaxis_title="Quantité", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# Tableau de juillet
july_row = item_data.iloc[0]
status = "✅ OK" if stock_july >= july_row["Predicted Quantity"] else "❌ Risque BO"
table = pd.DataFrame({
    "Mois": [july_row["Date"].strftime("%B %Y")],
    "Prévision": [round(july_row["Predicted Quantity"], 1)],
    "Stock disponible": [stock_july],
    "Statut": [status]
})

st.markdown("#### 📋 Détail premier mois (stock vs prévision)")
st.dataframe(table.style.applymap(lambda val: 'background-color: #d4edda; color: #155724' if val == '✅ OK' else 'background-color: #f8d7da; color: #721c24', subset=["Statut"]))

# Résumé final
st.markdown("### 🤖 Synthèse IA")
st.markdown(f"Le produit **{july_row['Item Description']}** est prévu à **{july_row['Predicted Quantity']:.1f} unités** en **{july_row['Date'].strftime('%B %Y')}**. Le stock est de **{stock_july}**. {'🟢 Pas de rupture prévue.' if status == '✅ OK' else '🔴 Attention, risque de rupture immédiate.'}")
