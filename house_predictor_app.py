import streamlit as st
import pandas as pd
import numpy as np
import joblib
import urllib.request
import os

import custom_transformers  # Important! Keeps your pipeline stable
from custom_transformers import BaseEstimator, TransformerMixin

# === Function to download model files from Google Drive ===
@st.cache_resource
def download_model_file(file_id, filename):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        urllib.request.urlretrieve(url, filename)
    return joblib.load(filename)

# === Google Drive File IDs ===
PIPELINE_FILE_ID = "1WJ-2CoZjeVN8_F1xlUbz6wvtiY_U_n14"
MODEL_FILE_ID = "1s5wrAlVq-rqZJKsxmZPWPKMW4pZ54eEX"

# === Load model and pipeline from Google Drive ===
pipeline = download_model_file(PIPELINE_FILE_ID, "full_pipeline.pkl")
model = download_model_file(MODEL_FILE_ID, "final_random_forest_model.pkl")

# === Streamlit App UI ===
st.title("🏡 House Price Predictor")
st.write("Enter house details below:")

# User input
longitude = st.number_input("Longitude", value=-122.0)
latitude = st.number_input("Latitude", value=37.0)
housing_median_age = st.number_input("Housing Median Age", value=20)
total_rooms = st.number_input("Total Rooms", value=1000)
total_bedrooms = st.number_input("Total Bedrooms", value=200)
population = st.number_input("Population", value=800)
households = st.number_input("Households", value=300)
median_income = st.number_input("Median Income", value=3.0)
ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

# Prediction
if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    input_prepared = pipeline.transform(input_df)
    prediction = model.predict(input_prepared)

    st.success(f"💰 Predicted House Price: ${prediction[0]:,.2f}")
