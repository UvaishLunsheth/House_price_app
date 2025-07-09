import streamlit as st
import pandas as pd
import joblib
import urllib.request
import os

# ✅ Import custom transformers BEFORE loading pipeline
import custom_transformers

@st.cache_resource  # Cache download for performance
def load_model_from_drive():
    url = "https://drive.google.com/uc?id=1s5wrAlVq-rqZJKsxmZPWPKMW4pZ54eEX"
    filename = "final_random_forest_model.pkl"

    if not os.path.exists(filename):
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(url, filename)

    return joblib.load(filename)

# Load model
model = load_model_from_drive()

# Load the model and pipeline
model = joblib.load("final_random_forest_model.pkl")
pipeline = joblib.load("full_pipeline.pkl")

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 California House Price Predictor")

st.markdown("Enter the features of a house to predict its median price:")

# Input fields
longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=41.0)
total_rooms = st.number_input("Total Rooms", value=880.0)
total_bedrooms = st.number_input("Total Bedrooms", value=129.0)
population = st.number_input("Population", value=322.0)
households = st.number_input("Households", value=126.0)
median_income = st.number_input("Median Income", value=8.3252)
ocean_proximity = st.selectbox("Ocean Proximity", ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

# Predict function
def predict(model, pipeline, data):
    df = pd.DataFrame([data])
    processed = pipeline.transform(df)
    prediction = model.predict(processed)
    return round(prediction[0], 2)

# Predict button
if st.button("Predict House Price"):
    input_data = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }

    price = predict(model, pipeline, input_data)
    st.success(f"🏡 Estimated Median House Value: ${price}")


