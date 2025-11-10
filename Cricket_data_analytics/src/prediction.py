# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 10:23:17 2025

@author: admin
"""

import os
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score


def train_models(A, b, name):
    A_scaled = scaler.fit_transform(A)

    if name == "Linear Regression":
        model = LinearRegression()
        model.fit(A_scaled)
        pred = scaler.inverse_transform(model.predict(A_scaled))