"""
data_preprocessing.py
----------------------
Handles loading, cleaning, feature engineering, encoding, scaling, and
train/test splitting of the Telco Customer Churn dataset.

Stages:
    1. load_data()              -> read raw CSV
    2. inspect_data()           -> diagnostic printout
    3. clean_data()              -> fix dtypes, handle missing values, drop junk columns
    4. add_engineered_features() -> create new predictive features
    5. encode_features()         -> label encode binary cols, one-hot encode multi-class cols
    6. scale_numeric_features()  -> standardize numeric columns
    7. prepare_train_test_data() -> orchestrates 4-6 + train/test split + saves artifacts
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Allow importing config.py from project root when this file is run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn CSV into a DataFrame.
    """
    df = pd.read_csv(path)
    print(f"[load_data] Loaded dataset with shape: {df.shape}")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Print a quick diagnostic summary of the dataset.
    """
    print("\n--- Shape ---")
    print(df.shape)

    print("\n--- Dtypes ---")
    print(df.dtypes)

    print("\n--- Missing values per column ---")
    print(df.isnull().sum())

    print("\n--- Duplicate rows ---")
    print(df.duplicated().sum())

    print("\n--- Target distribution ---")
    print(df[config.TARGET_COLUMN].value_counts(normalize=True))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Telco dataset:
      - Fix TotalCharges (object -> float, handle blank strings)
      - Drop customerID (identifier, no predictive value)
      - Encode target column Churn: Yes/No -> 1/0
      - Drop exact duplicate rows
    """
    df = df.copy()

    # --- Fix TotalCharges ---
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # --- Drop identifier column ---
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # --- Encode target column ---
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].map({"Yes": 1, "No": 0})

    # --- Drop exact duplicate rows ---
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    if before != after:
        print(f"[clean_data] Dropped {before - after} duplicate rows")

    print(f"[clean_data] Cleaned dataset shape: {df.shape}")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features that carry additional predictive signal
    beyond the raw columns.

    Adds:
        - tenure_group: bucketed tenure (captures early-churn pattern from EDA)
        - avg_monthly_spend: TotalCharges / tenure (guards against divide-by-zero)
    """
    df = df.copy()

    bins = [0, 12, 24, 48, np.inf]
    labels = ["0-1yr", "1-2yr", "2-4yr", "4yr+"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=True, include_lowest=True)

    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    print("[add_engineered_features] Added 'tenure_group' and 'avg_monthly_spend'")
    return df


def encode_features(df: pd.DataFrame, fit: bool = True, encoders: dict = None):
    """
    Encode categorical columns:
        - Binary categorical columns (2 unique values) -> Label Encoding
        - Multi-class nominal columns -> One-Hot Encoding
    """
    df = df.copy()
    encoders = encoders or {}

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    binary_cols = [col for col in categorical_cols if df[col].nunique() == 2]
    multiclass_cols = [col for col in categorical_cols if df[col].nunique() > 2]

    for col in binary_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col])

    df = pd.get_dummies(df, columns=multiclass_cols, drop_first=True)

    print(f"[encode_features] Binary (label-encoded): {binary_cols}")
    print(f"[encode_features] Multi-class (one-hot-encoded): {multiclass_cols}")

    return df, encoders


def scale_numeric_features(df: pd.DataFrame, numeric_cols: list, fit: bool = True, scaler: StandardScaler = None):
    """
    Standardize numeric columns (mean=0, std=1).
    """
    df = df.copy()
    scaler = scaler or StandardScaler()

    if fit:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, scaler


def prepare_train_test_data(df: pd.DataFrame):
    """
    Full pipeline: engineer features -> encode -> split -> scale.
    Saves fitted encoders/scaler/feature-names to disk for reuse at prediction time.
    """
    df = add_engineered_features(df)

    y = df[config.TARGET_COLUMN]
    X = df.drop(columns=[config.TARGET_COLUMN])

    X_encoded, encoders = encode_features(X, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend"]
    X_train, scaler = scale_numeric_features(X_train, numeric_cols, fit=True)
    X_test, _ = scale_numeric_features(X_test, numeric_cols, fit=False, scaler=scaler)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(encoders, config.ENCODER_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(X_train.columns.tolist(), config.FEATURE_NAMES_PATH)

    print(f"[prepare_train_test_data] X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"[prepare_train_test_data] Saved encoders, scaler, and feature names to /models")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    raw_df = load_data()
    inspect_data(raw_df)
    cleaned_df = clean_data(raw_df)

    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    cleaned_df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"[main] Saved cleaned data to {config.PROCESSED_DATA_PATH}")

    X_train, X_test, y_train, y_test = prepare_train_test_data(cleaned_df)
    print("\n--- Sample of X_train ---")
    print(X_train.head())