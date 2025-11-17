# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 12:04:31 2025

@author: aksha
"""

import streamlit as st
from src.load_data_model import load_data
import plotly.express as px

st.markdown("<style>.block-container{max-width:95%;}</style>", unsafe_allow_html=True)

st.header("Feature vs Target Analysis for batsman")

df = load_data()

player = st.selectbox("Select Player", sorted(df['batter'].unique()))

target = "BPI"
num_features = df.select_dtypes(include=['int64','float64']).columns.tolist()

selected_feature = st.selectbox(
    "Select a feature:",
    [f for f in num_features if f != target]
)

# Correlation
corr = df[selected_feature].corr(df[target])
st.write(f"**Correlation with BPI:** `{corr:.3f}`")

# Scatter plot
fig = px.scatter(
    df,
    x=selected_feature,
    y=target,
    hover_data=[player],
    trendline="ols",
    title=f"{selected_feature} vs BPI"
)
st.plotly_chart(fig, use_container_width=True)


