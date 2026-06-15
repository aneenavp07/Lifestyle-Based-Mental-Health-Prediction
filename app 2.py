import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model
model = load_model("wellbeing_model.h5")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Page config
st.set_page_config(page_title="Wellbeing Predictor", layout="centered")

# Title
st.markdown("<h1 style='text-align:center;'>🧠 Lifestyle-Based Mental Health Wellbeing Prediction</h1>", unsafe_allow_html=True)
st.write("")

# Inputs
age = st.slider("🎂 Age", 18, 35, 25)
gender = st.selectbox("👤 Gender", ["Male", "Female"])
sleep = st.slider("😴 Sleep Hours", 0, 12, 6)
stress = st.slider("😖 Stress Level", 1, 10, 5)

# Encode
gender_val = 0 if gender == "Male" else 1
input_data = np.array([[age, gender_val, stress, sleep]])
input_scaled = scaler.transform(input_data)

st.write("")

# Predict
if st.button("🔍 Predict Wellbeing"):
    pred = model.predict(input_scaled)
    pred_class = np.argmax(pred)
    result = le.inverse_transform([pred_class])[0]

    # FULL COLORED BOX UI
    if result == "Poor":
        st.markdown(f"""
        <div style="
            padding:30px;
            border-radius:15px;
            background-color:#ff4d4d;
            color:white;
            text-align:center;
            box-shadow:0px 6px 15px rgba(0,0,0,0.2);
        ">
            <h2>😞 Poor Wellbeing</h2>
            <p>Your lifestyle may be negatively affecting your mental health.</p>
        </div>
        """, unsafe_allow_html=True)

    elif result == "Average":
        st.markdown(f"""
        <div style="
            padding:30px;
            border-radius:15px;
            background-color:#ff9900;
            color:white;
            text-align:center;
            box-shadow:0px 6px 15px rgba(0,0,0,0.2);
        ">
            <h2>😐 Average Wellbeing</h2>
            <p>You are doing okay, but improving habits can boost your wellbeing.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div style="
            padding:30px;
            border-radius:15px;
            background-color:#28a745;
            color:white;
            text-align:center;
            box-shadow:0px 6px 15px rgba(0,0,0,0.2);
        ">
            <h2>😃 Good Wellbeing</h2>
            <p>Great! Your lifestyle supports good mental health.</p>
        </div>
        """, unsafe_allow_html=True)
