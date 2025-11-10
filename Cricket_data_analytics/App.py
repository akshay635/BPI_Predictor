import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from src.features_selector import select_features
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(page_title="🏏 Dynamic BPI Predictor", page_icon="🏏", layout="wide")

st.title("🏏 Dynamic IPL Batsman Performance Predictor (with Model & Feature Selection)")

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    return {
        "Decision Tree": joblib.load("C:/Users/aksha/Cricket_data_analytics/models/dt_model.joblib"),
        "Random Forest": joblib.load("C:/Users/aksha/Cricket_data_analytics/models/rf_model.joblib"),
        "XGBoost": joblib.load("C:/Users/aksha/Cricket_data_analytics/models/xgb_model.joblib")
    }

models = load_models()

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/aksha/Cricket_data_analytics/data/seasonal_stats.csv")

df = load_data()

scaler = joblib.load('C:/Users/aksha/Cricket_data_analytics/models/scaler.joblib')

# ---------------- Inputs ----------------
col1, col2 = st.columns(2)
with col1:
    player = st.selectbox("🎯 Select Player", sorted(df['batter'].unique()))
with col2:
    seasons = df[df['batter'] == player]['season'].unique()
    season = st.selectbox("📅 Select Season", sorted(seasons, reverse=True))

# ---------------- Model Selection ----------------
selected_model_name = st.selectbox(
    "🤖 Select Model for Prediction",
    ["Decision Tree", "Random Forest", "XGBoost"]
)

model = models[selected_model_name]
player_row = df[(df['batter'] == player) & (df['season'] == season)]

if player_row.empty:
    st.warning("No data found for this player and season.")
    st.stop()

# ---------------- Feature Selection ----------------
st.markdown("### 🔍 Feature Selection")
X = df.drop(columns=['BPI', 'batter', 'season', 'batting_team'], errors="ignore")
y = df['BPI']

selected_features, importance_df = select_features(X, y, selected_model_name, top_n=11)

st.write(f"Top 10 features selected for {selected_model_name}:")
st.dataframe(importance_df.head(10), use_container_width=True)

# ---------------- Prediction ----------------
st.markdown("---")
st.subheader(f"🎯 Predict BPI for {player} ({season}) using {selected_model_name}")

X_input_scaled = scaler.transform(player_row[selected_features].values)
prediction = model.predict(X_input_scaled)[0]

st.success(f"Predicted BPI: **{prediction:.2f}**")
st.success(f"Actual BPI: **{player_row['BPI'].values[0]:.2f}**")
st.success(f"Difference: **{abs(prediction - player_row['BPI'].values[0]):.2f}**")

if 'BPI' in player_row.columns:

    comparison_df = pd.DataFrame({
        "Player": [player],
        "Season": [season],
        "Actual BPI": [player_row['BPI'].values[0]],
        "Predicted BPI": [prediction],
        "Difference": [round(prediction - player_row['BPI'].values[0], 2)],
        "mean_absolute_error" : mean_absolute_error([prediction], [player_row['BPI'].values[0]]),
        "mean_squared_error" : mean_squared_error([prediction], [player_row['BPI'].values[0]]), 
        "rmse" : root_mean_squared_error([prediction], [player_row['BPI'].values[0]])
    })

    st.dataframe(comparison_df.style.highlight_max(subset=["Predicted BPI"], color='lightgreen'), use_container_width=True)


# ---------------- Visualization ----------------
fig = px.bar(
    importance_df.head(10),
    x="Feature",
    y="Score",
    title=f"Feature Importance / F-score ({selected_model_name})",
    text_auto=True
)
st.plotly_chart(fig, use_container_width=True)
