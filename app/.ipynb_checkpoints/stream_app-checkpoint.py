import streamlit as st
import pandas as pd
import joblib
import sys
import os

from src.data_prep import clean_data, group_rare_categories
from src.feature_engineering import new_features

project_root = os.path.abspath(r"C:\Users\letov\Projects\salary_prediction_project")
sys.path.append(project_root)

st.title("Salary Prediction App")
st.write("Predict whether an individual earns >50K or ≤50K.")

# Load model
model = joblib.load("models/salary_model.joblib")

# User inputs
age = st.number_input("Age", 18, 90, 30)
workclass = st.selectbox("Workclass", ["private", "self-emp", "gov", "unknown"])
education_num = st.number_input("Education Number", 1, 16, 10)
marital_status = st.selectbox("Marital Status", ["married", "single", "divorced"])
occupation = st.text_input("Occupation", "tech-support")
relationship = st.text_input("Relationship", "not-in-family")
race = st.text_input("Race", "white")
sex = st.selectbox("Sex", ["male", "female"])
capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
hours_per_week = st.number_input("Hours per Week", 1, 80, 40)
native_country = st.text_input("Native Country", "united-states")

# Build dataframe
input_df = pd.DataFrame([{
    "age": age,
    "workclass": workclass,
    "education-num": education_num,
    "marital-status": marital_status,
    "occupation": occupation,
    "relationship": relationship,
    "race": race,
    "sex": sex,
    "capital-gain": capital_gain,
    "capital-loss": capital_loss,
    "hours-per-week": hours_per_week,
    "native-country": native_country
}])

# Preprocess
df_clean = clean_data(input_df, drop_dupes=False)
df_grouped = group_rare_categories(df_clean)
df_features = new_features(df_grouped)

# Predict
if st.button("Predict Salary"):
    pred = model.predict(df_features)[0]
    label = ">50K" if pred == 1 else "≤50K"
    st.success(f"Predicted Salary Category: **{label}**")