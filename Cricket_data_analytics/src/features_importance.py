# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file."""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

def select_features(X, y, model_name, corr_threshold=0.95, top_n=11):
    """
    Dynamically performs feature selection based on the selected model.

    Returns:
        selected_features (list): Top N features.
        importance_df (pd.DataFrame): Ranked features with importance/score.
    """

    # Step 1: Remove highly correlated features
    corr_matrix = X.corr().abs()
    enable_corr_filter = st.checkbox("🧹 Enable Correlation Filtering (|r| > 0.95)", value=True)

    if enable_corr_filter:
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] >= corr_threshold)]

        if to_drop:
            st.warning(f"⚠️ Dropped {len(to_drop)} highly correlated features: {to_drop}")
            X_filtered = X.drop(columns=to_drop, errors='ignore')
        else:
            st.info("✅ No features found with correlation > 0.9")
        
        if model_name == "Random Forest":
            model = RandomForestRegressor(n_estimators=200, random_state=42)
        elif model_name == "XGBoost":
            model = XGBRegressor(n_estimators=200, random_state=42)
        else:
            model = DecisionTreeRegressor(random_state=42)
            
        model.fit(X_filtered, y)
        importance_df = pd.DataFrame({
                "Feature": X_filtered.columns,
                "Score": model.feature_importances_
        }).sort_values(by="Score", ascending=False)
    else:
        st.info("🧩 Correlation filtering is disabled. Using all features.")
        # Step 2: Feature ranking
        # Tree-based models (Random Forest / XGBoost)
        if model_name == "Random Forest":
            model = RandomForestRegressor(n_estimators=200, random_state=42)
        elif model_name == "XGBoost":
            model = XGBRegressor(n_estimators=200, random_state=42)
        else:
            model = DecisionTreeRegressor(random_state=42)
            
        model.fit(X, y)
        importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Score": model.feature_importances_
        }).sort_values(by="Score", ascending=False)

    selected_features = importance_df.head(top_n)["Feature"].tolist()
    
    return selected_features
