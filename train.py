from src.data_prep import load_data, clean_data, group_rare_categories
from src.feature_engineering import new_features
from src.model_training import split_data, build_model, train_model, save_model
from src.model_evaluation import evaluate_model
from src.logger import get_logger

import sys
import os

project_root = os.path.abspath(r"C:\Users\letov\Projects\salary_prediction_project")
sys.path.append(project_root)

logger = get_logger("TRAIN")

def main():
    logger.info("Starting training pipeline...")

    df = load_data(
        r"C:\Users\letov\Projects\salary_prediction_project\data\salary_prediction_classification.csv"
    )
    logger.info(f"Loaded data: {df.shape}")

    df = clean_data(df)
    logger.info(f"After clean_data: {df.shape}")

    df = group_rare_categories(df)
    logger.info(f"After group_rare_categories: {df.shape}")

    df = new_features(df)
    logger.info(f"After new_features: {df.shape}")

    X_train, X_test, y_train, y_test = split_data(df)
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    model = build_model(X_train, y_train)
    logger.info("Model built.")

    model = train_model(model, X_train, y_train)
    logger.info("Model training complete.")

    f1 = evaluate_model(model, X_test, y_test)
    logger.info(f"Final F1 score: {f1:.4f}")
    print(f"Final F1 score: {f1:.4f}")

    save_model(model, "models/salary_model.joblib")
    logger.info("Model saved to models/salary_model.joblib")

    logger.info("Training pipeline finished.")

if __name__ == "__main__":
    main()