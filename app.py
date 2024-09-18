# app.py
import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('model.sav', 'rb') as f:
    model = pickle.load(f)

# Function to predict AC sales
def predict_sales(temp, season):
    # Prepare input data
    season_encoded = [1 if season == 'Summer' else 0]
    input_data = np.array([temp] + season_encoded).reshape(1, -1)
    
    # Predict using the loaded model
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit UI
st.title("AC Sales Prediction Based on Season")

# Input fields
temperature = st.number_input("Enter the temperature (°C):", min_value=0, max_value=60, value=30)
season = st.selectbox("Select the season:", ('Summer', 'Winter'))

# Predict button
if st.button("Predict"):
    result = predict_sales(temperature, season)
    st.write(f"Predicted AC Sales: {int(result)} units")

