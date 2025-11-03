import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Cape Town Airbnb", layout="centered")
st.title("Cape Town Airbnb Price Predictor")
st.markdown("**XGBoost — The Best Model — Live Predictions**")

# --- LOAD MODEL & FEATURES ---
@st.cache_resource
def load_model():
    model = joblib.load("cape_town_model.pkl")
    features = joblib.load("listings_features.pkl")
    return model, features

model, feature_cols = load_model()

# --- LOAD UI DATA ---
@st.cache_data
def load_ui_data():
    df = pd.read_csv("data/sample_listings.csv")
    return df

df = load_ui_data()

# --- USER INPUT ---
st.sidebar.header("Enter Listing Details")

guests = st.sidebar.slider("Guests", 1, 16, 4)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 2)
bathrooms = st.sidebar.slider("Bathrooms", 0.5, 6.0, 1.0, 0.5)
room_type = st.sidebar.selectbox("Room Type", df['room_type'].unique())
area = st.sidebar.selectbox("Area", df['neighbourhood_cleansed'].unique())

# --- BUILD INPUT SAFELY ---
input_dict = {
    'accommodates': guests,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'room_type': room_type,
    'neighbourhood_cleansed': area
}

# Create full input with ALL model features
inp = pd.DataFrame([input_dict])

# Add missing columns with 0
for col in feature_cols:
    if col not in inp.columns:
        inp[col] = 0

# Reorder to match model
inp = inp[feature_cols]

# --- PREDICT ---
if st.sidebar.button("Predict Price"):
    pred_log = model.predict(inp)[0]
    price = np.expm1(pred_log)
    st.success(f"**R {price:,.0f} per night**")
    st.balloons()

st.caption("**Model: XGBoost (Tuned)** | Data: Inside Airbnb | Live on Streamlit")