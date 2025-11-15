# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 10:47:15 2025

@author: aksha
"""
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.load_data_model import load_data
from sklearn.decomposition import PCA
import plotly.express as px

# No set_page_config() here
st.markdown("<style>.block-container{max-width:95%;}</style>", unsafe_allow_html=True)

st.title("Clustering players based on the performance in each season")

df = load_data()

print(df.columns)

df['runs/inning'] = round((df['total_runs']/df['matches']), 2)
df['balls/inning'] = round((df['total_balls']/df['matches']), 2)
df['not_outs%'] = round((df['Not_outs']/df['matches'])*100, 2)

features = ['runs/inning', 'balls/inning', 'not_outs%', 'strike_rate',
            'boundary_runs(%)', 'dot_balls(%)', 'BPI']


scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=7, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

cluster_order = df.groupby('cluster')['BPI'].mean().sort_values().index.tolist()

performance_labels = [
    "Worst Season",
    "Below Average Season",
    "Average Season",
    "Above Average Season",
    "Good Season",
    "Consistent Season",
    "Extraordinary Season"
]

mapping = {cluster_order[i]: performance_labels[i] for i in range(7)}
df['performance_cluster'] = df['cluster'].map(mapping)

pca = PCA(n_components=2)
components = pca.fit_transform(df_scaled)
df['PC1'] = components[:,0]
df['PC2'] = components[:,1]

fig = px.scatter(
    df,
    x="PC1",
    y="PC2",
    color="performance_cluster",
    hover_data=['batter', 'season'] + features,
    title="Player Performance Clusters"
)

st.plotly_chart(fig, use_container_width=True)

