# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 08:36:44 2025

@author: admin
"""

import pandas as pd
from src.data_cleaning import load_and_clean_data
from src.feature_engineering import add_features
from src.prediction import load_model, predict_next_season

def test_full_pipeline():
    df = load_and_clean_data("data/season_stats.csv")
    assert not df.empty, "Dataset should not be empty"

    df = add_features(df)
    assert "boundary_pct" in df.columns, "Feature engineering failed"

    model = load_model()
    assert model is not None, "Model not loaded properly"

    features = ["total_runs", "batting_average", "strike_rate"]
    sample = df[features].iloc[0].values
    prediction = predict_next_season(model, sample)

    assert len(prediction) == 3, "Prediction should contain 3 values"
    assert all(p >= 0 for p in prediction), "Predicted values should be non-negative"

    print("\n✅ Integration test successful!")

