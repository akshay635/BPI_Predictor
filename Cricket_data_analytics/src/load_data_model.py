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
    df = pd.read_csv('Cricket_data_analytics/data/seasonal_stats.csv')
    return df

@st.cache_resource
def load_models():
    #decision tree regressor
    dt_model =  joblib.load("Cricket_data_analytics/models/dt_model.joblib")
    #random forest regressor
    rf_model =  joblib.load("Cricket_data_analytics/models/rf_model.joblib")
    #XGBoost regressor
    models = {'Decision Tree': dt_model, 'Random Forest': rf_model, 'XGBoost': xgb.XGBRegressor()}
    models['XGBoost'].load_model("Cricket_data_analytics/models/xgb_model.json")
    #StandardScaler
    scaler = joblib.load("Cricket_data_analytics/models/scaler.joblib")
    return models, scaler

@st.cache_data
def train_test_split(df):
    X = df.drop(columns=['season', 'batting_team', 'batter', 'BPI'])
    y = df['BPI']
    
    return X, y


