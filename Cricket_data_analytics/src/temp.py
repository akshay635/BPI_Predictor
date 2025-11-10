# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file."""
# ipl_dashboard_dash.py

import pandas as pd
from sklearn.prepreprocessing import MinMaxScaler, RobustScaler, StandardScaler

def scaling_data(df):
    minmaxscaler = MinMaxScaler()
    standardscaler = StandardScaler()
    robustscaler = RobustScaler()
    standard_columns = ['matches', 'total_balls', 'fours', 'sixes', 'dismissals', 'dot_balls', 'ones', 'twos', 'threes']
    robust_columns = ['total_runs']
    minmax_columns = ['strike_rate', 'batting_average', 'thirty_plus', 'fifty_plus', 'hundred_plus',
                      'boundaries_runs', 'runs/ball', 'runs_rbw', 'runs_rbw%', 'boundary_runs%', 'dot_balls%']

    df[standard_columns] = standardscaler.fit_transform(df)
    df[robust_columns] = robustscaler.fit_transform(df)
    df[minmaxscaler] = minmaxscaler.fit_transform(df)

    return df
