# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 11:09:40 2025

@author: admin
"""

import pandas as pd
from src.data_cleaning import load_and_clean_data

def test_data_cleaning_basic():
    df = pd.DataFrame({
        "batter": ["A", "B", None],
        "season": [2021, 2022, 2023],
        "total_runs": [500, 600, None],
        "batting_average": [50, 40, 35],
        "strike_rate": [130, 120, None]
    })

    df.to_csv("data/temp_test.csv", index=False)
    cleaned = load_and_clean_data("data/temp_test.csv")

    # Checks
    assert "batter" in cleaned.columns
    assert cleaned.isnull().sum().sum() == 0  # no missing values
    assert len(cleaned) < len(df)  # dropped some rows
