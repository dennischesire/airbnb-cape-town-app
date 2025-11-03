import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Cape Town Airbnb", layout="centered")
st.title("Cape Town Airbnb Price Predictor")
st.markdown("**XGBoost — The Best Model — Live Predictions**")

# --- LOAD MODEL & FEATURE LIST ---
@st.cache_resource
def load_model():
    model = joblib.load("cape_town_model.pkl")
    feature_cols = joblib.load("listings_features.pkl")
    return model, feature_cols

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

# --- BUILD INPUT ---
input_dict = {
    'accommodates': guests,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'room_type': room_type,
    'neighbourhood_cleansed': area
}

inp = pd.DataFrame([input_dict])

# --- REBUILD PREPROCESSING (Same as Training) ---
numeric_features = ['accommodates', 'bedrooms', 'bathrooms']
categorical_features = ['room_type', 'neighbourhood_cleansed']

# One-hot encode categoricals
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(inp[categorical_features])
cat_cols = encoder.get_feature_names_out(categorical_features)

# Combine numeric + encoded categoricals
numeric_part = inp[numeric_features].values
full_features = np.hstack([numeric_part, encoded_cats])
feature_names = numeric_features + list(cat_cols)

# Create DataFrame with correct feature names
inp_processed = pd.DataFrame(full_features, columns=feature_names)

# --- ALIGN WITH MODEL FEATURES ---
for col in feature_cols:
    if col not in inp_processed.columns:
        inp_processed[col] = 0

inp_final = inp_processed[feature_cols]  # Exact match

# --- PREDICT ---
if st.sidebar.button("Predict Price"):
    pred_log = model.predict(inp_final)[0]
    price = np.expm1(pred_log)
    st.success(f"**R {price:,.0f} per night**")
    st.balloons()

st.caption("**Model: XGBoost** | Real Preprocessing | Live on Streamlit")