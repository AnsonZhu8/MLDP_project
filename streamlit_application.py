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
    "Age",
    min_value=20,
    max_value=80,
    value=50
)

# Chest Pain Type
chest_pain = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic"
    ]
)

# Convert to model values
chest_pain_dict = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
}

chest_pain = chest_pain_dict[chest_pain]

# Resting Blood Pressure
resting_bp = st.number_input(
    "Resting Blood Pressure (mmHg)",
    min_value=0,
    max_value=250,
    value=120
)

# Cholesterol
cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    min_value=0,
    max_value=700,
    value=200
)

# Resting ECG
resting_ecg = st.selectbox(
    "Resting ECG Result",
    [
        "Normal",
        "ST-T Wave Abnormality",
        "Left Ventricular Hypertrophy"
    ]
)

# Convert to model values
resting_ecg_dict = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

resting_ecg = resting_ecg_dict[resting_ecg]

# Maximum Heart Rate
max_hr = st.slider(
    "Maximum Heart Rate Achieved",
    min_value=60,
    max_value=202,
    value=140
)

# Oldpeak
oldpeak = st.number_input(
    "Oldpeak (ST Depression)",
    min_value=-2.6,
    max_value=6.2,
    value=1.0,
    step=0.1
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