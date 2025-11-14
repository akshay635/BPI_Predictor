# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:49:38 2025

@author: aksha
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.load_data_model import load_data, load_models, train_test_split
from src.features_selector import select_features
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# No set_page_config() here
st.markdown("<style>.block-container{max-width:95%;}</style>", unsafe_allow_html=True)

st.title("📊 Model Evaluation — Actual vs Predicted BPI")

# ---------------- Load Dataset ----------------

df = load_data()

# ---------------- Load Models -----------------

models, scaler = load_models()

selected_model = st.selectbox("Select Model", list(models.keys()))

model = models[selected_model]

X, y = train_test_split(df)

selected_features, importance_df = select_features(X, y, model, top_n=11)

X_input_scaled = scaler.transform(X[selected_features].values)

y_pred = model.predict(X_input_scaled)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    "Player": df["batter"],
    "Season": df["season"],
    "Actual BPI": y,
    "Predicted BPI": y_pred
})
# Scatter plot
fig = px.scatter(
    comparison_df,
    x="Actual BPI", y="Predicted BPI",
    color="Season",
    hover_data=["Player"],
    title="Actual vs Predicted BPI (All Players)"
)
fig.add_shape(
    type="line",
    x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
    line=dict(color="red", dash="dash")
)
st.plotly_chart(fig, use_container_width=True)

# Metrics
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.3f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")
