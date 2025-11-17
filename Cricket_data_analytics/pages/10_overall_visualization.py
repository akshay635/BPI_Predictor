# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 11:39:58 2025

@author: aksha
"""

import streamlit as st
from src.load_data_model import load_data
import plotly.express as px

st.markdown("<style>.block-container{max-width:95%;}</style>", unsafe_allow_html=True)

st.title("Top 10 batters based on features")

df = load_data()

overall_stats = df.groupby('batter').agg(
    total_matches = ('matches', 'sum'), total_balls = ('total_balls', 'sum'),
    total_runs = ('total_runs', 'sum'), dot_balls = ('dot_balls', 'sum'),
    singles = ('ones', 'sum'), doubles = ('twos', 'sum'),
    triples = ('threes', 'sum'), fours = ('fours', 'sum'),
    sixes = ('sixes', 'sum'), strike_rate = ('strike_rate', 'mean'),
    batting_average = ('batting_average', 'mean'), thirties_plus = ('thirty_plus', 'sum'),
    fifties_plus = ('fifty_plus', 'sum'), hundred_plus = ('hundred_plus', 'sum'),
    dismissals = ('dismissals', 'sum'), not_outs = ('Not_outs', 'sum'),
    runs_4s_percent = ('runs_fours(4s)%', 'mean'), runs_6s_percent = ('runs_sixes(6s)%', 'mean'),
    boundary_runs_percent = ('boundary_runs(%)', 'mean'), dot_balls_percent = ('dot_balls(%)', 'mean'),
    Hussey_Index = ('Hussey_Index', 'mean'), DPPI = ('DPPI', 'mean'), BPI = ('BPI', 'mean')
).reset_index()

overall_stats = overall_stats.round(decimals=2)
overall_stats = overall_stats.query('total_matches >= 10')
features = overall_stats.select_dtypes(include=['int64', 'float64'])

feature = st.selectbox('Select feature', sorted(features))

top_20_batters = overall_stats.sort_values(by=feature, ascending=False)

fig = px.bar(top_20_batters.head(20),
             x="batter", y=feature,
             orientation="v",
             title=f"🏆 Top 20 Batters overall ({feature})",
             text=feature,
             color=feature,
             color_continuous_scale="viridis"
             )

fig.update_layout(height=600, template="simple_white")
fig.update_traces(textposition="outside")

st.plotly_chart(fig, use_container_width=True)


