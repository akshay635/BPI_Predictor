# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file."""


import streamlit as st
import pandas as pd
import plotly.express as px
#import plotly.objects as go
from src.load_data_model import load_data, load_models, train_test_split
from src.features_importance import select_features

# No set_page_config() here
st.markdown("<style>.block-container{max-width:95%;}</style>", unsafe_allow_html=True)

st.title("** Predicted BPI vs Actual BPI **")

df = load_data()

models, scaler = load_models()

X, y = train_test_split(df)

player = st.selectbox("Select a player", sorted(df['batter'].unique()))

selected_model = selected_model_name = st.selectbox(
    "🤖 Select Model for Prediction",
    ["Decision Tree", "Random Forest", "XGBoost"]
)

player_data = df[df['batter'] == player]
model = models[selected_model]

selected_features = select_features(X, y, model, top_n=11)
X_scaled = scaler.transform(player_data[selected_features].values)

y_pred = model.predict(X_scaled)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    "Player": player_data["batter"],
    "Season": player_data['season'],
    "Actual_BPI": y,
    "Predicted_BPI": y_pred
}, index=player_data.index)

# Scatter plot
fig = px.scatter(
    comparison_df,
    x="Actual_BPI", y="Predicted_BPI",
    hover_data=["Player"],
    title= f"Actual vs Predicted BPI ({player}) using scatter plot",
    trendline="ols"
)

st.plotly_chart(fig, use_container_width=True)


