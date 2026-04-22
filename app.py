import streamlit as st
import numpy as np
import pandas as pd
import pickle

# -------------------------------------------------------
# LOAD MODEL & ENCODERS
# -------------------------------------------------------
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

mood_mapping = {
    "happy": 1,
    "relaxed": 2,
    "neutral": 3,
    "tired": 5,
    "anxious": 7,
    "sad": 8,
    "angry": 9,
    "stressed": 10
}
# -------------------------------------------------------
# PAGE CONFIG (Modern UI)
# -------------------------------------------------------
st.set_page_config(page_title="Stress Predictor", layout="centered")

st.title("Student Stress Prediction App")
st.markdown("Predict stress level based on lifestyle habits")
# -------------------------------------------------------
# SAFE ENCODER FUNCTION (PREVENTS UNSEEN LABEL ERROR)
# -------------------------------------------------------
def safe_transform(encoder, value):
    value = str(value).lower().strip()
    classes = [c.lower() for c in encoder.classes_]

    if value in classes:
        index = classes.index(value)
        return encoder.transform([encoder.classes_[index]])[0]
    else:
        return encoder.transform([encoder.classes_[0]])[0]  # fallback
# -------------------------------------------------------
# USER INPUT
# -------------------------------------------------------
age = st.slider("Age", 15, 90, 40)
gender = st.selectbox("Gender", encoders["Gender"].classes_)
mood = st.selectbox("Mood", list(mood_mapping.keys()))
social_media = st.selectbox("Social Media Usage", encoders["Social_Media_Usage"].classes_)

screen_hours = st.slider("Screen Time (hours)", 1, 10, 5)
sleep_hours = st.slider("Sleep Hours", 3, 10, 7)

# -------------------------------------------------------
# ENCODE INPUT
# -------------------------------------------------------
gender_encoded = safe_transform(encoders["Gender"], gender)
mood_encoded = mood_mapping[mood]
social_encoded = safe_transform(encoders["Social_Media_Usage"], social_media)

# -------------------------------------------------------
# PREDICTION BUTTON
# -------------------------------------------------------
if st.button("Predict Stress Level"):
    # -------------------------------------------
    # MODEL PREDICTION BASED ON INPUT
    # -------------------------------------------
    interaction = mood_encoded * sleep_hours

    input_data = pd.DataFrame([{
        "Mood": mood_encoded,
        "screen_hours": screen_hours,
        "sleep_hours": sleep_hours,
        "Gender": gender_encoded,
        "Age": age,
        "Social_Media_Usage": social_encoded,
        "mood_sleep_interaction": interaction
    }])

    base_prediction = float(model.predict(input_data)[0])

    # -------------------------------------------
    # RULE-BASED ADJUSTMENTS (REALISTIC LOGIC)
    # -------------------------------------------
    stress_score = base_prediction

    # Adjust for sleep — poor sleep increases stress sharply
    if sleep_hours <= 4:
        stress_score += 2
    elif 5 <= sleep_hours < 7:
        stress_score += 1
    elif sleep_hours >= 8:
        stress_score -= 1

    # Adjust for screen time — excessive screen time raises stress slightly
    if screen_hours >= 8:
        stress_score += 1
    elif 1 <= screen_hours <= 3:
        stress_score -= 1

    # Gender differences (slight variance based on studies)
    gender_lower = gender.lower()
    if "female" in gender_lower:
        stress_score += 0.3  # slightly higher average reported stress
    elif "male" in gender_lower:
        stress_score -= 0.2

    # Social media usage factor — heavy usage increases stress
    sm = social_media.lower()
    if "high" in sm or "very often" in sm:
        stress_score += 1
    elif "moderate" in sm:
        stress_score += 0.3
    elif "low" in sm or "rarely" in sm:
        stress_score -= 0.2

    # Clamp stress_score between 1 and 5
    stress_score = max(1, min(5, round(stress_score, 2)))

    # -------------------------------------------
    # STRESS LEVEL COLOR FEEDBACK
    # -------------------------------------------
    color = "green"
    status = "Very Low Stress"
    if stress_score >= 4.5:
        color, status = "red", "Extremely High Stress ⚠️"
    elif stress_score >= 3.5:
        color, status = "orange", "High Stress 😟"
    elif stress_score >= 2.5:
        color, status = "gold", "Moderate Stress 🙂"
    elif stress_score >= 1.5:
        color, status = "lightgreen", "Low Stress 😊"

    # -------------------------------------------
    # FINAL OUTPUT
    # -------------------------------------------
    st.markdown(f"<h3 style='color:{color};'>Predicted Stress Level: {stress_score} ({status})</h3>", unsafe_allow_html=True)

    # -------------------------------------------
    # INPUT SUMMARY CHART
    # -------------------------------------------
    st.subheader("Input Summary")
    chart_data = pd.DataFrame({
        "Feature": ["Screen Time", "Sleep Hours", "Age"],
        "Value": [screen_hours, sleep_hours, age]
    })
    st.bar_chart(chart_data.set_index("Feature"))