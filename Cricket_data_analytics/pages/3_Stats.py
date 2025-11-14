# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file."""
# ipl_dashboard_dash.py

import streamlit as st
import pandas as pd
from src.load_data_model import load_data

# No set_page_config() here
st.markdown("<style>.block-container{max-width:95%;}</style>", unsafe_allow_html=True)

st.title("Player stats (Overall and Seasonwise))")

df = load_data()

player_selected = st.selectbox("🎯 Select Player", sorted(df['batter'].unique()))

player_stats = df[df['batter'] == player_selected]

if player_stats.empty:
    st.warning("No data found for this player and season.")
    st.stop()

player_stats = player_stats[['season', 'matches', 'total_runs',
                             'total_balls', 'batting_average',
                             'strike_rate','dot_balls', 'ones', 
                             'twos', 'fours', 'sixes','thirty_plus', 
                             'fifty_plus', 'hundred_plus', 'dismissals',
                             'Not_outs', 'Hussey_Index', 'DPPI', 'BPI']]

def add_overall_row(group):
    overall = pd.DataFrame({
        'season': ['Overall'],
        'matches': [group['matches'].sum()],
        'total_runs': [group['total_runs'].sum()],
        'total_balls': [group['total_balls'].sum()],
        'batting_average': [round(group['batting_average'].mean(), 2)],
        'strike_rate': [round(group['strike_rate'].mean(), 2)],
        'dot_balls': [group['dot_balls'].sum()],
        'ones': [group['ones'].sum()],
        'twos': [group['twos'].sum()],
        'fours': [group['fours'].sum()],
        'sixes': [group['sixes'].sum()],
        'thirty_plus':[group['thirty_plus'].sum()],
        'fifty_plus': [group['fifty_plus'].sum()], 
        'hundred_plus': [group['hundred_plus'].sum()], 
        'dismissals': [group['dismissals'].sum()],
        'Not_outs': [group['Not_outs'].sum()],
        'Hussey_Index':[round(group['Hussey_Index'].mean(), 2)],
        'DPPI': [round(group['DPPI'].mean(), 2)],
        'BPI': [round(group['BPI'].mean(), 2)]
    })
    return pd.concat([group, overall], ignore_index=True)

seasonal_with_overall = player_stats.groupby(df['batter'], group_keys=False).apply(add_overall_row)

def highlight_overall(row):
    return ['background-color: lightgreen; font-weight: bold;' if row['season'] == 'Overall' else '' for _ in row]

# Function to highlight a specific column
def highlight_column(s, col_name, color='lightgreen'):
    return [
        f'background-color: {color}' if s.name == col_name else ''
        for _ in s
    ]

# Apply styling to highlight the 'Score' column
styled_df = player_stats.style.apply(highlight_column, col_name='BPI', axis=0)

if st.button('Show both season-wise and overall stats'):
    st.title(f'Both Overall and Seasonwise stats of {player_selected}')

    st.dataframe(seasonal_with_overall.style.apply(highlight_overall, axis=1))
