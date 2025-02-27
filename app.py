import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
with open('newweather.pkl', 'rb') as f:
    model = pickle.load(f)

# Set up page configuration
st.set_page_config(
    page_icon='⛅',
    page_title="Weather Insight Engine"
)

# Application title and description
st.markdown("<h1>⛅ Weather Predictor: Forecasting Future Conditions</h1>", unsafe_allow_html=True)

# Input features
season = st.selectbox("Select Season", ["Choose option...", "Winter", "Spring", "Summer", "Autumn"])
humidity = st.number_input("**Humidity (%)**", min_value=0, max_value=100)
wind_speed = st.number_input("**Wind Speed (km/h)**", min_value=0.0, max_value=20.0)
precipitation = st.number_input("**Precipitation (chance %)**", min_value=0.0, max_value=100.0)
temperature = st.slider("**Temperature (°C)**", min_value=-100, max_value=100)

# Encode categorical variable
season_map = {"Winter": 3, "Spring": 1, "Summer": 2, "Autumn": 0}
season_encoded = season_map.get(season, -1)

# Mapping for prediction output to weather labels
weather_condition_map = {1: "🌧️Rainy", 0: "☁️Cloudy", 3: " ☀️Sunny", 2: "❄️Snowy"}

# Check if all required inputs are selected
if st.button("Make Prediction"):
    if season_encoded == -1:
        st.warning("Please select an option for the season.")
    else:
        # Prepare features for prediction, excluding 'Weather Type', 'Atmospheric Pressure', 'Location'
        features = np.array([
            temperature, humidity, wind_speed, precipitation, season_encoded
        ]).reshape(1, -1)

        # Define the feature names as expected by the model
        feature_names = ["Temperature", "Humidity", "Wind Speed", "Precipitation (%)", "Season"]
        features_df = pd.DataFrame(features, columns=feature_names)

        # Make prediction
        prediction_result = model.predict(features_df)

        # Map the prediction result to a weather condition label
        predicted_condition = weather_condition_map.get(prediction_result[0], "Unknown")

        # Display prediction result
        st.write("Prediction Result: Predicted Weather Condition:", predicted_condition)


