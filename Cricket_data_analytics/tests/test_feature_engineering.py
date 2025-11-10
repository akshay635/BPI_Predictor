# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 11:10:43 2025

@author: admin
"""

import pandas as pd
from src.feature_engineering import add_features

def test_add_features():
    df = pd.DataFrame({
        "fours": [10, 5],
        "sixes": [5, 2],
        "total_runs": [70, 40]
    })

    new_df = add_features(df)

    assert "boundary_pct" in new_df.columns
    assert all(new_df["boundary_pct"] >= 0)
    assert all(new_df["boundary_pct"] <= 100)
