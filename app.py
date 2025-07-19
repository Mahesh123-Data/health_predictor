import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Diabetes Risk Predictor 🧬")
st.write("Enter your medical details to know the risk prediction")

# Input fields
preg = st.number_input("Pregnancies", 0)
gluc = st.number_input("Glucose", 0)
bp = st.number_input("Blood Pressure", 0)
skin = st.number_input("Skin Thickness", 0)
insulin = st.number_input("Insulin", 0)
bmi = st.number_input("BMI", 0.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0)
age = st.number_input("Age", 0)

if st.button("Predict"):
    features = np.array([[preg, gluc, bp, skin, insulin, bmi, dpf, age]])
    features_scaled = scaler.transform(features)
    result = model.predict(features_scaled)[0]

    if result == 1:
        st.error("❗ You are likely at risk of diabetes.")
    else:
        st.success("✅ You are likely NOT at risk of diabetes.")
