import sys
import os

# Resolve project root (the folder that contains src/)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import joblib

from src.data_prep import clean_data, group_rare_categories
from src.feature_engineering import new_features
from src.logger import get_logger


# ----------------------------------------------------
# Initialize Logger
# ----------------------------------------------------
logger = get_logger("streamlit_app")
logger.info("Streamlit app started")
logger.info(f"ROOT_DIR resolved as: {ROOT_DIR}")


# ----------------------------------------------------
# UI Header
# ----------------------------------------------------
st.title("Salary Prediction App")
st.write("Predict whether an individual earns >50K or ≤50K.")
logger.info("UI loaded successfully")


# ----------------------------------------------------
# Load Model
# ----------------------------------------------------
MODEL_PATH = os.path.join(ROOT_DIR, "models", "salary_model.joblib")

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    st.error("Error loading model. Check logs.")
    st.stop()


# ----------------------------------------------------
# User Inputs
# ----------------------------------------------------
age = st.number_input("Age", 18, 90, 30)
workclass = st.selectbox("workclass", [
    "self-emp-not-inc", "local-gov", "state-gov", "private",
    "self-emp-inc", "federal-gov", "without-pay", "unknown"
])
education_num = st.number_input("Education Number", 1, 16, 10)
marital_status = st.selectbox("Marital Status", [
    "married-civ-spouse", "never-married", "separated", "widowed",
    "married-spouse-absent", "married-af-spouse", "divorced"
])
occupation = st.text_input("Occupation", "tech-support")
relationship = st.text_input("Relationship", "not-in-family")
race = st.text_input("Race", "white")
sex = st.selectbox("Sex", ["male", "female"])
capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
hours_per_week = st.number_input("Hours per Week", 1, 80, 40)
native_country = st.text_input("Native Country", "united-states")

logger.info("User input collected")


# ----------------------------------------------------
# Build DataFrame
# ----------------------------------------------------
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

logger.info(f"Input DataFrame created: {input_df.to_dict(orient='records')}")


# ----------------------------------------------------
# Preprocess
# ----------------------------------------------------
try:
    df_clean = clean_data(input_df)
    df_grouped = group_rare_categories(df_clean)
    df_features = new_features(df_grouped)

    logger.info(f"Preprocessing complete. Final shape: {df_features.shape}")

    # ----------------------------------------------------
    # Monitoring: Input Drift + Feature Health
    # ----------------------------------------------------
    try:
        # Numeric drift monitoring
        stats = df_features.describe().to_dict()
        logger.info(f"Feature summary stats: {stats}")

        # Categorical drift monitoring
        cat_summary = input_df.select_dtypes(include=['object']).apply(
            lambda x: x.value_counts().to_dict()
        )
        logger.info(f"Categorical value counts: {cat_summary.to_dict()}")

    except Exception as e:
        logger.error(f"Monitoring stats failed: {e}")

except Exception as e:
    logger.error(f"Preprocessing failed: {e}")
    st.error("Error during preprocessing. Check logs.")
    st.stop()


# ----------------------------------------------------
# Predict
# ----------------------------------------------------
if st.button("Predict Salary"):
    try:
        pred = model.predict(df_features)[0]
        label = ">50K" if pred == 1 else "≤50K"

        # Monitoring: prediction drift
        logger.info(f"Prediction raw value: {pred}")
        logger.info(f"Prediction label: {label}")

        st.success(f"Predicted Salary Category: **{label}**")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        st.error("Prediction error. Check logs.")