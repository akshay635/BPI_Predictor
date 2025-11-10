# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 10:55:33 2025

@author: admin
"""

import numpy as np
from src.prediction import load_model, predict_next_season

def test_model_prediction():
    model = load_model()
    sample_input = np.array([500, 45, 140])  # Runs, Avg, SR
    result = predict_next_season(model, sample_input)
    assert len(result) == 3  # Should predict 3 target metrics
