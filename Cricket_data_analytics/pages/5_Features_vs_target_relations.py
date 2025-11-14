# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 12:04:31 2025

@author: aksha
"""

import streamlit as st
import seaborn as sns
from src.load_data_model import load_data
import plotly.express as px

# No set_page_config() here
st.markdown("<style>.block-container{max-width:95%;}</style>", unsafe_allow_html=True)

st.title("Feature vs Target Analysis")

df = load_data()

#player = st.selectbox("Select Player", sorted(df['batter'].unique()))

target = "BPI"
num_features = df.select_dtypes(include=['int64','float64']).columns.tolist()

selected_feature = st.selectbox(
    "Select a feature:",
    [f for f in num_features if f != target]
)

tab1, tab2, tab3, tab4 = st.tabs(["Scatter Plot", "Distribution", 'Pair Plot', 'Reg Plot'])

# Correlation
corr = df[selected_feature].corr(df[target])
st.text(f"**Correlation with BPI:** `{corr:.3f}`")

#Scatter plot
with tab1:
    st.header("Plotly scatterplot in Streamlit")
    fig = px.scatter(df, x=selected_feature, y=target, trendline="ols", 
                     title=f"{selected_feature} vs BPI using scatter plot")
    st.plotly_chart(fig, use_container_width=True)

#bar plot
with tab2:
    st.header("Seaborn histplot in Streamlit")
    fig = sns.histplot(df[selected_feature], kde=True, bins=30)
    st.pyplot(fig.get_figure(), use_container_width=True)
    
with tab3:
    st.header("Seaborn pairplot in Streamlit")
    fig = sns.pairplot(df, x_vars=selected_feature, y_vars=target)
    st.pyplot(fig.fig, use_container_width=True)
    
with tab4:
    st.header("Seaborn regplot in Streamlit")

    # Create the plot
    sns.regplot(x=selected_feature, y=target, data=df)
    
    # Display in Streamlit
    st.pyplot(fig.fig, use_container_width=True)