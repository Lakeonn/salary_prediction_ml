import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

from src.data_prep import load_data, clean_data, group_rare_categories
from src.feature_engineering import new_features, build_preprocessor


"""---------------------------------------------------------------"""

def split_data(df):
    """Split into train/test."""
    X = df.drop("salary", axis=1)
    y = df["salary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    return X_train, X_test, y_train, y_test

"""---------------------------------------------------------------"""

def compute_class_weight(y):
    """Compute imbalance ratio (optional)."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return neg / pos

"""---------------------------------------------------------------"""

def build_model(X_train, y_train):
    """Create preprocessing + Random Forest model pipeline."""
    preprocessor = build_preprocessor(X_train)

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced",
            n_jobs=-1,
            random_state=123
        ))
    ])

    return model

"""---------------------------------------------------------------"""

def train_model(model, X_train, y_train):
    """Fit the model."""
    model.fit(X_train, y_train)
    return model

"""---------------------------------------------------------------"""

def save_model(model, path="models/salary_model.joblib"):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")