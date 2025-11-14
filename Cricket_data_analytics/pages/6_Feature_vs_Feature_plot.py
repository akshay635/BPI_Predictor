# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 00:43:07 2025

@author: aksha
"""

import streamlit as st
from src.load_data_model import load_data
import plotly.express as px
import statsmodels.api as sm 

# No set_page_config() here
st.markdown("<style>.block-container{max-width:95%;}</style>", unsafe_allow_html=True)

st.title("Feature vs Feature Analysis")

df = load_data()

num_features = df.select_dtypes(include=['int64','float64']).columns.tolist()

features1 = st.selectbox('Select first feature', sorted(num_features))
features2 = st.selectbox('Select second feature', sorted(num_features))

fig = px.scatter(df, x=features1, y=features2, trendline="ols", 
                 title=f"{features1} vs {features2} using scatter plot")

st.plotly_chart(fig, use_container_width=True)



