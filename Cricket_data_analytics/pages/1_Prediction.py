# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:46:41 2025

@author: aksha
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.features_selector import select_features
from src.load_data_model import load_data, load_models, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

# No set_page_config() here
st.markdown("<style>.block-container{max-width:95%;}</style>", unsafe_allow_html=True)


st.title("⚙️ Player BPI Prediction")

# ---------------- Load Dataset ----------------

df = load_data()

# ---------------- Load Models -----------------

models, scaler = load_models()

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
X, y = train_test_split(df)

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