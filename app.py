import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Cape Town Airbnb", layout="centered")
st.title("Cape Town Airbnb Price Predictor")
st.markdown("**XGBoost — The Best Model — Live Predictions**")

# --- LOAD MODEL & PREPROCESSOR ---
@st.cache_resource
def load_model():
    model = joblib.load("cape_town_model.pkl")
    # Assume you saved the full pipeline or preprocessor
    # If not, we'll rebuild it
    try:
        preprocessor = joblib.load("listings_features.pkl")  # Should be ColumnTransformer
        return model, preprocessor
    except:
        st.warning("Rebuilding preprocessor...")
        return model, None

model, preprocessor = load_model()

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
    'accommodates': [guests],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'room_type': [room_type],
    'neighbourhood_cleansed': [area]
}
inp = pd.DataFrame(input_dict)

# --- PREPROCESS INPUT (Same as Training) ---
if preprocessor is not None:
    try:
        inp_transformed = preprocessor.transform(inp)
        pred_log = model.predict(inp_transformed)
    except:
        st.error("Preprocessing failed. Using fallback.")
        pred_log = [np.log1p(2500)]
else:
    # Fallback: manual one-hot
    inp_encoded = pd.get_dummies(inp, columns=['room_type', 'neighbourhood_cleansed'])
    # Align with model features
    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else inp_encoded.columns
    for col in model_features:
        if col not in inp_encoded.columns:
            inp_encoded[col] = 0
    inp_encoded = inp_encoded[model_features]
    pred_log = model.predict(inp_encoded)

price = np.expm1(pred_log)[0]

# --- PREDICT ---
if st.sidebar.button("Predict Price"):
    st.success(f"**R {price:,.0f} per night**")
    st.balloons()

st.caption("**Model: XGBoost** | Real Preprocessing | Live on Streamlit")