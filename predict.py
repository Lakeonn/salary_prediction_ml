import pandas as pd
import joblib
import os

from src.data_prep import clean_data, group_rare_categories
from src.feature_engineering import new_features
from src.utils import print_section, Timer
from src.logger import get_logger


def load_model(model_path="models/salary_model.joblib"):
    """Load the trained salary prediction model."""
    logger = get_logger("salary_predictor")
    logger.info(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
    return model


def prepare_data(df):
    """Apply the same preprocessing steps used during training."""
    logger = get_logger("salary_predictor")
    logger.info("Starting preprocessing pipeline")

    df = clean_data(df)
    logger.info(f"After clean_data: {df.shape}")

    df = group_rare_categories(df)
    logger.info(f"After group_rare_categories: {df.shape}")

    df = new_features(df)
    logger.info(f"After new_features: {df.shape}")

    logger.info("Preprocessing complete")
    return df


def predict(model, df):
    """Generate salary predictions."""
    logger = get_logger("salary_predictor")
    logger.info("Running model.predict()")

    preds = model.predict(df)
    logger.info(f"Generated {len(preds)} predictions")

    return preds


def main(input_path="data/new_inference_data.csv", output_path="salary_predictions.csv"):

    logger = get_logger("salary_predictor")

    print_section("Salary Prediction Pipeline")
    logger.info("Starting salary prediction pipeline")

    # ----------------------------------------------------
    # 1. Load Model
    # ----------------------------------------------------
    print_section("Loading Model")
    with Timer("Model load time"):
        model = load_model()

    # ----------------------------------------------------
    # 2. Load New Data
    # ----------------------------------------------------
    print_section("Loading Input Data")
    logger.info(f"Reading input CSV: {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded input data with shape: {df.shape}")

    if "salary" in df.columns:
        logger.warning("Input contains 'salary' column — dropping it.")
        df = df.drop(columns=["salary"])

    # ----------------------------------------------------
    # 3. Preprocess Data
    # ----------------------------------------------------
    print_section("Preprocessing Data")
    with Timer("Preprocessing time"):
        df_prepared = prepare_data(df)

    logger.info(f"Prepared data shape: {df_prepared.shape}")

    # ----------------------------------------------------
    # 4. Predict
    # ----------------------------------------------------
    print_section("Generating Predictions")
    logger.info("Beginning prediction step")

    with Timer("Prediction time"):
        preds = predict(model, df_prepared)

    # ----------------------------------------------------
    # 5. Save Output
    # ----------------------------------------------------
    print_section("Saving Predictions")
    logger.info(f"Saving predictions to: {output_path}")

    output_df = df.copy()
    output_df["predicted_salary"] = preds

    output_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved successfully to {output_path}")

    print_section("Prediction Pipeline Complete")
    logger.info("Salary prediction pipeline completed successfully")


if __name__ == "__main__":
    main()