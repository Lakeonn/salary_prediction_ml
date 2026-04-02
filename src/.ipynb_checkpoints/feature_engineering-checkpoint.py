import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

"""---------------------------------------------------------------"""

def new_features(df):
    df = df.copy()

    # Select numeric columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Exclude salary if present
    numerical_cols = [col for col in numerical_cols if col != "salary"]

    # Define meaningful interaction pairs
    important_pairs = [
        ("age", "hours-per-week"),
        ("education-num", "capital-gain"),
        ("education-num", "hours-per-week"),
    ]

    # Create interaction features only if both columns exist
    for col1, col2 in important_pairs:
        if col1 in df.columns and col2 in df.columns:
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]

    return df

"""---------------------------------------------------------------"""

def build_preprocessor(df):
    # Identify column types
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Build the transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    return preprocessor