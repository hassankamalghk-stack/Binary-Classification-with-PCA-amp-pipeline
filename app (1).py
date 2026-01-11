import streamlit as st
import numpy as np
import joblib

# Load saved objects
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

st.title("Diabetes Prediction App 🩺")
st.write("Enter patient medical details to predict diabetes")

# User Inputs
preg = st.number_input("Pregnancies", 0, 20, 1)
glu = st.number_input("Glucose", 0, 300, 120)
bp = st.number_input("Blood Pressure", 0, 200, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
ins = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):
    input_data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)

    prediction = model.predict(input_pca)[0]

    if prediction == 1:
        st.error("⚠️ Diabetic (1)")
    else:
        st.success("✅ Non-Diabetic (0)")
