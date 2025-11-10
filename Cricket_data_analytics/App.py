import streamlit as st
import pandas as pd
from src.data_cleaning import clean_data
from src.feature_engineering import create_features
from src.prediction import load_model, predict_next_season
from src.utils import plot_predictions

st.set_page_config(page_title="🏏 IPL Player Predictor", layout="wide")
st.title("🏏 IPL Player Performance Prediction Dashboard")

@st.cache_resource
def get_model():
    return load_model("models/ipl_rf_model.joblib")

model = get_model()

uploaded_file = st.file_uploader("📂 Upload seasonwise.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = clean_data(df)
    df = create_features(df)

    st.sidebar.header("Player & Season Selection")
    player_choice = st.sidebar.selectbox("Select Player", sorted(df["batter"].unique()))
    season_choice = st.sidebar.selectbox("Select Season", sorted(df["season"].unique()))

    player_data = df[(df["batter"] == player_choice) & (df["season"] == season_choice)]

    if not player_data.empty:
        pred = predict_next_season(model, player_data)
        st.subheader(f"Predicted Next Season Stats for {player_choice}")
        st.json(pred)
        fig = plot_predictions(player_choice, pred)
        st.pyplot(fig)
    else:
        st.warning("No data found for the selected player and season.")
else:
    st.info("👆 Upload the `seasonwise.csv` file to begin.")
