# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 07:55:02 2025

@author: aksha
"""
import streamlit as st
from src.load_data_model import load_data
import plotly.express as px

st.markdown("<style>.block-container{max-width:95%;}</style>", unsafe_allow_html=True)

st.title("Top 10 batters based on features")

df = load_data()

features = df.drop(columns=['season', 'batting_team', 'batter']).columns.tolist()

season = st.selectbox("Select a season", sorted(df['season'].unique()))

feature = st.selectbox('Select feature', sorted(features))

season_stats = df[df['season'] == season]

top_20_batters = season_stats.sort_values(by=feature, ascending=False)

top_20_batters = top_20_batters.round(decimals=2)

fig = px.bar(top_20_batters.head(20),
             x="batter", y=feature,
             orientation="v",
             title=f"🏆 Top 20 Batters in {season} ({feature})",
             text=feature,
             color=feature,
             color_continuous_scale="viridis"
             )

fig.update_layout(height=600, template="simple_white")
fig.update_traces(textposition="outside")

st.plotly_chart(fig, use_container_width=True)

