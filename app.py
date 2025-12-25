import streamlit as st
import numpy as np
import pandas as pd
import pickle


with open("heart_random_forest_model.pkl", "rb") as file:
    rf_model = pickle.load(file)



st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease risk.")

age = st.number_input("Age", min_value=1, max_value=120, value=50)

sex = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp_text = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)
cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
cp = cp_map[cp_text]

bp = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol Level", value=200)

fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
fbs = 1 if fbs == "Yes" else 0

ekg_text = st.selectbox(
    "EKG Result",
    ["Normal", "ST-T Abnormality", "LV Hypertrophy"]
)
ekg_map = {
    "Normal": 0,
    "ST-T Abnormality": 1,
    "LV Hypertrophy": 2
}
ekg = ekg_map[ekg_text]

thalach = st.number_input("Maximum Heart Rate Achieved", value=150)

exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang = 1 if exang == "Yes" else 0

oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)

slope_text = st.selectbox(
    "Slope of ST Segment",
    ["Upsloping", "Flat", "Downsloping"]
)
slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}
slope = slope_map[slope_text]

ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])

thal_text = st.selectbox(
    "Thallium Test Result",
    ["Normal", "Fixed Defect", "Reversible Defect"]
)
thal_map = {
    "Normal": 3,
    "Fixed Defect": 6,
    "Reversible Defect": 7
}
thal = thal_map[thal_text]

if st.button("Predict Heart Disease"):

    user_input = np.array([[age, sex, cp, bp, chol, fbs,
                            ekg, thalach, exang,
                            oldpeak, slope, ca, thal]])

    prediction = rf_model.predict(user_input)
    probability = rf_model.predict_proba(user_input)

    presence_index = list(rf_model.classes_).index("Presence")
    disease_prob = probability[0][presence_index] * 100

    st.subheader("Result")

    if disease_prob >= 50:
        st.error(f"⚠️ Heart Disease Likely\n\nRisk Probability: {disease_prob:.2f}%")
    else:
        st.success(f"✅ Heart Disease Unlikely\n\nRisk Probability: {disease_prob:.2f}%")


    st.caption("⚠️ This tool is for educational purposes only.")

