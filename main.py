import streamlit as st
import pandas as pd
import numpy as np
import os
from Source_Code import load_weather_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Set page config
st.set_page_config(
    page_title="Weather Condition Predictor",
    page_icon="🌤️",
    layout="wide"
)

# Title and description
st.title("🌤️ Weather Condition Predictor")
st.markdown("""
This application predicts weather conditions based on various meteorological parameters.
Enter the weather parameters below to get a prediction.
""")

# Load the model and data
@st.cache_data
def load_model_and_data():
    try:
        # Load the data
        df = load_weather_data()
        
        # Prepare the model
        feature_columns = [
            'temp', 'humidity', 'precip', 'windspeed', 'cloudcover', 
            'sealevelpressure', 'solarradiation', 'uvindex',
            'month', 'day', 'hour'
        ]
        
        # Encode categorical target variable
        le = LabelEncoder()
        y = le.fit_transform(df['conditions'])
        
        # Prepare features
        X = df[feature_columns]
        
        # Train the model
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        return rf, le, feature_columns, None
    except Exception as e:
        return None, None, None, str(e)

# Load model and data
model, label_encoder, feature_columns, error = load_model_and_data()

if error:
    st.error(f"""
    ⚠️ Error loading the model: {error}
    
    Please make sure:
    1. The datasets are in the correct location
    2. All required packages are installed
    3. You have proper permissions to access the files
    """)
    st.stop()

# Create input form
st.subheader("Input Parameters")
col1, col2 = st.columns(2)

with col1:
    temp = st.number_input("Temperature (°C)", -50.0, 50.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
    precip = st.number_input("Precipitation (mm)", 0.0, 100.0, 0.0)
    windspeed = st.number_input("Wind Speed (km/h)", 0.0, 200.0, 10.0)
    cloudcover = st.number_input("Cloud Cover (%)", 0.0, 100.0, 50.0)

with col2:
    sealevelpressure = st.number_input("Sea Level Pressure (hPa)", 900.0, 1100.0, 1013.0)
    solarradiation = st.number_input("Solar Radiation (W/m²)", 0.0, 1000.0, 500.0)
    uvindex = st.number_input("UV Index", 0.0, 12.0, 5.0)
    month = st.number_input("Month", 1, 12, 6)
    day = st.number_input("Day", 1, 31, 15)
    hour = st.number_input("Hour", 0, 23, 12)

# Create prediction button
if st.button("Predict Weather Condition"):
    # Prepare input data
    input_data = pd.DataFrame({
        'temp': [temp],
        'humidity': [humidity],
        'precip': [precip],
        'windspeed': [windspeed],
        'cloudcover': [cloudcover],
        'sealevelpressure': [sealevelpressure],
        'solarradiation': [solarradiation],
        'uvindex': [uvindex],
        'month': [month],
        'day': [day],
        'hour': [hour]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    predicted_condition = label_encoder.inverse_transform(prediction)[0]
    
    # Display prediction
    st.subheader("Prediction Result")
    st.success(f"Predicted Weather Condition: {predicted_condition}")
    
    # Display feature importances
    st.subheader("Feature Importances")
    importances = pd.Series(model.feature_importances_, index=feature_columns)
    importances = importances.sort_values(ascending=False)
    
    # Create a bar chart of feature importances
    st.bar_chart(importances)

# Add some information about the model
st.sidebar.title("About")
st.sidebar.info("""
This application uses a Random Forest Classifier to predict weather conditions based on various meteorological parameters.

The model was trained on historical weather data from Hanoi, Vietnam, and can predict weather conditions such as:
- Clear
- Partly Cloudy
- Cloudy
- Rain
- And more...

The prediction is based on the following parameters:
- Temperature
- Humidity
- Precipitation
- Wind Speed
- Cloud Cover
- Sea Level Pressure
- Solar Radiation
- UV Index
- Time-based features (Month, Day, Hour)
""") 