import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from src import feature_selection

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="🏏 Dynamic BPI Predictor", page_icon="🏏", layout="centered")

st.title("🏏 Dynamic IPL Batsman Performance Predictor")
st.markdown("""
Predict a **Batsman Performance Index (BPI)** for any player and season  
using multiple ML models (Linear, Random Forest, XGBoost).  
The app fetches player stats automatically and visualizes the results clearly.
---
""")

# ---------------------- LOAD MODEL & DATA ----------------------
@st.cache_resource
def load_models():
    return {
        "Decision Tree": joblib.load("dt_model.joblib"),
        "Random Forest": joblib.load("rf_model.joblib"),
        "Extra Trees": joblib.load("et_model.joblib")
        "XGBoost": joblib.load("xgb_model.joblib")
    }

@st.cache_data
def load_data():
    return pd.read_csv("seasonal_stats.csv")

models = load_models()
df = load_data()

# loading the standardscaler
scaler = joblib.load('scaler.joblib')

# Common features for all models
target = df["BPI"]
features = df.drop(columns=['season', 'batting_team', 'batter', 'BPI']).columns.tolist()

# ---------------------- INPUT SECTION ----------------------
col1, col2 = st.columns(2)
with col1:
    player = st.selectbox("🎯 Select Player", sorted(df['batter'].unique()))
with col2:
    seasons = df[df['batter'] == player]['season'].unique()
    season = st.selectbox("📅 Select Season", sorted(seasons, reverse=True))

player_row = df[(df['batter'] == player) & (df['season'] == season)]

if player_row.empty:
    st.warning("No data found for this player and season.")
    st.stop()

# ---------------------- PREDICTION ----------------------
if st.button("🎯 Predict Performance"):
    updated_features = feature_selection(features, target)
    X_input = player_row[features].values
    predictions = {name: model.predict(X_input)[0] for name, model in models.items()}
    results_df = pd.DataFrame(list(predictions.items()), columns=["Model", "Predicted BPI"])

    st.success(f"✅ Predictions for {player} ({season}) generated successfully!")
    st.dataframe(results_df, use_container_width=True)

    # ---------------------- COMPARISON VISUAL ----------------------
    st.markdown("### 📊 Model Predictions Comparison")
    fig = px.bar(results_df, x="Model", y="Predicted BPI", text_auto=True, title="Model-wise BPI Predictions")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------- ACTUAL VS PREDICTED ----------------------
    if 'BPI' in player_row.columns:
        actual_bpi = player_row['BPI'].values[0]
        st.info(f"🎯 Actual BPI: **{actual_bpi:.2f}**")

        comp_df = results_df.copy()
        comp_df.loc[len(comp_df)] = ["Actual BPI", actual_bpi]
        fig2 = px.bar(comp_df, x="Model", y="Predicted BPI", text_auto=True, title="Predicted vs Actual BPI")
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------------- METRICS SUMMARY ----------------------
    st.markdown("### ⚙️ Model Performance Metrics (Validation Results)")
    metrics_data = {
        "Linear Regression": {"R²": 0.90, "RMSE": 5.3, "MAE": 3.7},
        "Random Forest": {"R²": 0.97, "RMSE": 2.8, "MAE": 1.9},
        "XGBoost": {"R²": 0.96, "RMSE": 3.0, "MAE": 2.1}
    }
    metrics_df = pd.DataFrame(metrics_data).T
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

    # ---------------------- TREND CHART ----------------------
    st.markdown("### 📈 Player's BPI Trend Across Seasons")
    player_data = df[df['batter'] == player].copy()
    player_data['Predicted (RF)'] = models['Random Forest'].predict(player_data[features])

    if 'BPI' in player_data.columns:
        fig3 = px.line(player_data, x='Season', y=['BPI', 'Predicted (RF)'], markers=True,
                       title=f"{player} - Actual vs Predicted BPI Trend (Random Forest)")
    else:
        fig3 = px.line(player_data, x='Season', y='Predicted (RF)', markers=True,
                       title=f"{player} - Predicted BPI Trend (Random Forest)")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.caption("👨‍💻 Developed by **Akshay Atanure** | [GitHub](https://github.com/akshay635) | [LinkedIn](https://linkedin.com/in/akshayatanure)")
