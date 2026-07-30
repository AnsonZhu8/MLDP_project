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
    layout="wide"
)

## Title
st.title("❤️ Heart Disease Prediction")
st.write(
    "Enter the patient's health information below to predict the likelihood of heart disease."
)

st.divider()

## User Inputs

left, right = st.columns([3, 2])

with left:

    # Age
    age = st.slider(
        "👤 Age",
        min_value=20,
        max_value=80,
        value=50
    )

    # Chest Pain Type
    chest_pain = st.selectbox(
        "🫀 Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-Anginal Pain",
            "Asymptomatic"
        ]
    )

    # Description shown below
    chest_pain_desc = {
        "Typical Angina": "Chest pain related to the heart.",
        "Atypical Angina": "Chest pain not related to the heart.",
        "Non-Anginal Pain": "Sharp and non-continuous. pain.",
        "Asymptomatic": "No chest pain or noticeable symptoms."
    }

    st.caption(f"ℹ️ {chest_pain_desc[chest_pain]}")

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
        "🩺 Resting Blood Pressure (mmHg)",
        min_value=50,
        max_value=250,
        value=120
    )

    # Cholesterol
    cholesterol = st.number_input(
        "🧪 Cholesterol (mg/dL)",
        min_value=80,
        max_value=700,
        value=200
    )

    # Resting ECG
    resting_ecg = st.selectbox(
        "📈 Resting ECG Result",
        [
            "Normal",
            "Minor Electrical Changes",
            "Thickened Heart Muscle"
        ]
    )

    # Description
    resting_ecg_desc = {
        "Normal": "ℹ️ The heart's electrical activity appears normal.",
        "Minor Electrical Changes": "ℹ️ Small changes in the heart's electrical activity",
        "Thickened Heart Muscle": "ℹ️ The heart's main pumping muscle is thicker than normal, often due to high blood pressure or other heart conditions."
    }

    st.caption(resting_ecg_desc[resting_ecg])

    # Convert to model values
    resting_ecg_dict = {
        "Normal": 0,
        "Minor Electrical Changes": 1,
        "Thickened Heart Muscle": 2
    }

    resting_ecg = resting_ecg_dict[resting_ecg]

    # Maximum Heart Rate
    max_hr = st.slider(
        "❤️ Maximum Heart Rate Achieved During Exercise",
        min_value=60,
        max_value=202,
        value=140
    )

    # Oldpeak
    oldpeak = st.number_input(
        "📉 Heart Stress During Exercise (Oldpeak)",
        min_value=-2.6,
        max_value=6.2,
        value=1.0,
        step=0.1
    )

    st.caption(
        "ℹ️ Measures how much the heart shows signs of stress during exercise. "
        "Higher values may indicate a greater risk of heart disease."
    )



if predict:

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

    if resting_bp < 50:
        st.error("Resting blood pressure must be between 50 and 250 mmHg.")
        st.stop()

    if cholesterol < 80:
        st.error("Cholesterol must be at least 80 mg/dL.")
        st.stop()

    # Make prediction
    try:
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

        with right:

            st.divider()

            predict = st.button(
            "🫀 Predict Heart Disease",
            use_container_width=True
            )

            st.subheader("🩺 Prediction Result")

            if prediction == 1:
                st.error("⚠️ High Risk of Heart Disease")
            else:
                st.success("✅ Low Risk of Heart Disease")

            st.divider()

            st.subheader("📊 Prediction Probability")

            st.write(f"**Probability of Heart Disease:** {probability_yes:.2%}")
            st.progress(float(probability_yes))

            st.write(f"**Probability of No Heart Disease:** {probability_no:.2%}")
            st.progress(float(probability_no))
    except Exception as e:
        st.error(f"An error occurred while predicting: {e}")