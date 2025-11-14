# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 12:04:31 2025

@author: aksha
"""
#import pandas as pd
#import numpy as np
#from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import streamlit as st
from src.load_data_model import load_data
import statsmodels.api as sm 

# No set_page_config() here
st.markdown("<style>.block-container{max-width:95%;}</style>", unsafe_allow_html=True)

st.title("Comparing 2 players with each other")

df = load_data()

player1 = st.selectbox("Select Player 1", sorted(df['batter'].unique()))
player2 = st.selectbox("Select Player 2", sorted(df['batter'].unique()))

num_features = df.select_dtypes(include=['int64','float64']).columns.tolist()

feature = st.selectbox('Select a feature for comparison', sorted(num_features))

player1_stats = df[df['batter'] == player1]
player2_stats = df[df['batter'] == player2]

seasons1 = player1_stats['season'].tolist()
seasons2 = player2_stats['season'].tolist()

fig = go.Figure(data=[
    go.Bar(name=player1, x=seasons1, y=player1_stats[feature]),
    go.Bar(name=player2, x=seasons2, y=player2_stats[feature])
])

fig.update_layout(barmode='group', title="Player Comparison")

st.plotly_chart(fig, use_container_width=True)
