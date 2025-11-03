import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Cape Town Airbnb", layout="centered")
st.title("Cape Town Airbnb Price Predictor")
st.markdown("**XGBoost — The Best Model — Live & Accurate**")

# --- LOAD MODEL & FEATURES ---
@st.cache_resource
def load_model():
    model = joblib.load("cape_town_model.pkl")
    feature_cols = joblib.load("listings_features.pkl")  # List of final columns
    return model, feature_cols

model, model_features = load_model()

# --- LOAD UI DATA ---
@st.cache_data
def load_ui_data():
    df = pd.read_csv("data/sample_listings.csv")
    return df

ui_data = load_ui_data()

# --- USER INPUT ---
st.sidebar.header("Enter Listing Details")
guests = st.sidebar.slider("Guests", 1, 16, 4)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 2)
bathrooms = st.sidebar.slider("Bathrooms", 0.5, 6.0, 1.0, 0.5)
room_type = st.sidebar.selectbox("Room Type", ui_data['room_type'].unique())
area = st.sidebar.selectbox("Area", ui_data['neighbourhood_cleansed'].unique())

# --- BUILD RAW INPUT ---
input_raw = pd.DataFrame([{
    'accommodates': guests,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'room_type': room_type,
    'neighbourhood_cleansed': area
}])

# --- ONE-HOT ENCODE (MATCH TRAINING) ---
# Extract categories from UI data (same as training)
room_cats = sorted(ui_data['room_type'].unique())
area_cats = sorted(ui_data['neighbourhood_cleansed'].unique())

# Create encoder with known categories
encoder = OneHotEncoder(categories=[room_cats, area_cats], sparse_output=False, handle_unknown='ignore')
encoder.fit(ui_data[['room_type', 'neighbourhood_cleansed']])

# Transform input
encoded = encoder.transform(input_raw[['room_type', 'neighbourhood_cleansed']])
encoded_cols = encoder.get_feature_names_out(['room_type', 'neighbourhood_cleansed'])

# Combine numeric + encoded
numeric = input_raw[['accommodates', 'bedrooms', 'bathrooms']].values
full_array = np.hstack([numeric, encoded])
full_cols = ['accommodates', 'bedrooms', 'bathrooms'] + list(encoded_cols)

inp_final = pd.DataFrame(full_array, columns=full_cols)

# --- ALIGN WITH MODEL FEATURES ---
for col in model_features:
    if col not in inp_final.columns:
        inp_final[col] = 0
inp_final = inp_final[model_features]  # Exact order

# --- PREDICT ---
if st.sidebar.button("Predict Price"):
    pred = model.predict(inp_final)[0]
    price = np.expm1(pred)
    st.success(f"**R {price:,.0f} per night**")
    st.balloons()

st.caption("**XGBoost + One-Hot Encoding** | Real Model | Live on Streamlit")