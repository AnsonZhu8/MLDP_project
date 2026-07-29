import joblib
import streamlit as st
import numpy as np
import pandas as pd

## Load trained model
model = joblib.load("mldp_project_model.pkl")

## Streamlit page settings
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

## Title
st.title("❤️ Heart Disease Prediction")
st.write(
    "Enter the patient's health information below to predict the likelihood of heart disease."
)

st.divider()

## User Inputs

# Age
age = st.slider(
    "👤 Age",
    ...
)

# Chest Pain
chest_pain = st.selectbox(
    "🫀 Chest Pain Type",
    ...
)

# Blood Pressure
resting_bp = st.number_input(
    "🩺 Resting Blood Pressure (mmHg)",
    ...
)

# Cholesterol
cholesterol = st.number_input(
    "🧪 Cholesterol (mg/dL)",
    ...
)

# ECG
resting_ecg = st.selectbox(
    "📈 Resting ECG Result",
    ...
)

# Heart Rate
max_hr = st.slider(
    "❤️ Maximum Heart Rate Achieved",
    ...
)

# Oldpeak
oldpeak = st.number_input(
    "📉 Oldpeak (ST Depression)",
    ...
)

st.divider()

if st.button("🔍 Predict Heart Disease"):

    # Create DataFrame
    df_input = pd.DataFrame({
        "Age": [age],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "Oldpeak": [oldpeak]
    })

    # Make prediction
    prediction = model.predict(df_input)[0]

    # Prediction probabilities
    probability = model.predict_proba(df_input)[0]

    probability_no = probability[0]
    probability_yes = probability[1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.subheader("Prediction Probability")

    st.write(f"**Probability of Heart Disease:** {probability_yes:.2%}")
    st.progress(float(probability_yes))

    st.write(f"**Probability of No Heart Disease:** {probability_no:.2%}")
    st.progress(float(probability_no))