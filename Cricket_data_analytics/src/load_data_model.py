# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 16:51:42 2025

@author: aksha
"""

import pandas as pd
import joblib
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv('C:/Users/aksha/Cricket_data_analytics/data/seasonal_stats.csv')
    return df

@st.cache_resource
def load_models():
    models = {
        "Decision Tree": joblib.load("C:/Users/aksha/Cricket_data_analytics/models/dt_model.joblib"),
        "Random Forest": joblib.load("C:/Users/aksha/Cricket_data_analytics/models/rf_model.joblib"),
        "XGBoost": joblib.load("C:/Users/aksha/Cricket_data_analytics/models/xgb_model.joblib")
    }

    scaler = joblib.load("C:/Users/aksha/Cricket_data_analytics/models/scaler.joblib")
    return models, scaler

@st.cache_data
def train_test_split(df):
    X = df.drop(columns=['season', 'batting_team', 'batter', 'BPI'])
    y = df['BPI']
    
    return X, y



