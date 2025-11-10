# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 10:24:29 2025

@author: admin
"""

import matplotlib.pyplot as plt
import streamlit as st

def plot_performance(actual, predicted, labels):
    """Bar chart of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(labels, actual, alpha=0.6, label="Actual", color="blue")
    ax.bar(labels, predicted, alpha=0.6, label="Predicted", color="orange")
    ax.legend()
    ax.set_title("Actual vs Predicted Stats")
    st.pyplot(fig)
