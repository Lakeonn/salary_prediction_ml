import pandas as pd

def load_data(path: str):
    df = pd.read_csv(path)
    return df

"""---------------------------------------------------------------"""

def group_rare_categories(df, top_n=10):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        top_values = df[col].value_counts().nlargest(top_n).index
        df[col] = df[col].apply(lambda x: x if x in top_values else 'Other')

    return df

"""---------------------------------------------------------------"""

def clean_data(df):
    df = df.copy()
    # df = df.drop_duplicates()
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for columns in categorical_cols:
        df[columns] = df[columns].str.strip().replace({"?":"Unknown"})
        df[columns] = df[columns].str.strip().str.lower()

    df[categorical_cols] = df[categorical_cols].fillna("unknown")
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    df = df.drop(['fnlwgt', 'education'], axis=1, errors='ignore')
    
    if 'salary' in df.columns:
        df['salary'] = df['salary'].map({'<=50k': 0, '>50k': 1, 'unknown': 0})

    return df


