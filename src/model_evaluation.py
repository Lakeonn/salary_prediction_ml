import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from src.data_prep import load_data, clean_data, group_rare_categories
from src.feature_engineering import new_features, build_preprocessor
from src.model_training import split_data, compute_class_weight, build_model, train_model


def evaluate_model(model, X_test, y_test):
    """Evaluate F1 score."""
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"F1 score: {f1:.4f}")
    return f1
