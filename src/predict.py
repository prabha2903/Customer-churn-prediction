"""
predict.py
----------
Loads the trained model and preprocessing artifacts (encoders, scaler,
feature names) to make predictions on new, unseen customer data — both
single-customer (from a Streamlit form) and bulk (from an uploaded CSV).

CRITICAL: This module never re-fits any encoder/scaler. It only applies
the exact transformations learned during training, to guarantee the model
sees data in the same shape/distribution it was trained on.
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_preprocessing import add_engineered_features


def load_artifacts():
    """
    Load the trained model and all preprocessing artifacts saved during
    training. Called once when the Streamlit app starts.

    Returns
    -------
    dict with keys: model, encoders, scaler, feature_names
    """
    artifacts = {
        "model": joblib.load(config.BEST_MODEL_PATH),
        "encoders": joblib.load(config.ENCODER_PATH),
        "scaler": joblib.load(config.SCALER_PATH),
        "feature_names": joblib.load(config.FEATURE_NAMES_PATH),
    }
    print("[load_artifacts] Model and preprocessing artifacts loaded successfully")
    return artifacts


def _transform_input(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Core shared transformation logic for both single and bulk prediction.
    Applies: feature engineering -> label encoding (fitted) -> one-hot
    encoding -> column alignment -> scaling (fitted).

    This is a private helper (leading underscore) — not meant to be called
    directly from the Streamlit app; use predict_single/predict_bulk instead.
    """
    df = df.copy()
    encoders = artifacts["encoders"]
    scaler = artifacts["scaler"]
    feature_names = artifacts["feature_names"]

    # --- Step 1: Feature engineering (same function used in training) ---
    df = add_engineered_features(df)

    # --- Step 2: Label encode binary columns using FITTED encoders ---
    for col, encoder in encoders.items():
        if col in df.columns:
            # Guard against unseen categories crashing LabelEncoder.transform()
            known_classes = set(encoder.classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else encoder.classes_[0])
            df[col] = encoder.transform(df[col])

    # --- Step 3: One-hot encode remaining multi-class categorical columns ---
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # --- Step 4: Align columns to match training-time feature set exactly ---
    # Any column the model saw in training but is missing here (e.g., a
    # dummy category not present in this batch) gets added with value 0.
    # Any extra column not seen in training gets dropped.
    df = df.reindex(columns=feature_names, fill_value=0)

    # --- Step 5: Scale numeric columns using the FITTED scaler ---
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


def _risk_level(probability: float) -> str:
    """
    Convert a raw churn probability into a business-friendly risk bucket.

    Thresholds:
        < 0.35        -> Low
        0.35 - 0.65    -> Medium
        > 0.65        -> High
    """
    if probability < 0.35:
        return "Low"
    elif probability < 0.65:
        return "Medium"
    else:
        return "High"


def predict_single(customer_data: dict, artifacts: dict) -> dict:
    """
    Predict churn for a single customer, provided as a dict of raw
    (unencoded, unscaled) feature values — exactly as a Streamlit form
    would collect them.

    Parameters
    ----------
    customer_data : dict
        e.g. {"gender": "Female", "tenure": 5, "MonthlyCharges": 70.35, ...}
    artifacts : dict
        Output of load_artifacts()

    Returns
    -------
    dict with keys: prediction (0/1), churn_probability, risk_level
    """
    df = pd.DataFrame([customer_data])
    transformed = _transform_input(df, artifacts)

    model = artifacts["model"]
    probability = model.predict_proba(transformed)[0, 1]
    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "churn_probability": round(float(probability), 4),
        "risk_level": _risk_level(probability),
    }


def predict_bulk(df_raw: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Predict churn for a batch of customers, provided as a raw DataFrame
    (e.g., from an uploaded CSV) with the same columns as the original
    Telco dataset (minus the target column).

    Returns
    -------
    pd.DataFrame
        Original data with three new columns appended:
        Prediction, Churn_Probability, Risk_Level
    """
    df_raw = df_raw.copy()

    # Drop target/ID columns if accidentally present in the uploaded CSV
    for col in ["customerID", config.TARGET_COLUMN]:
        if col in df_raw.columns:
            df_raw = df_raw.drop(columns=[col])

    transformed = _transform_input(df_raw, artifacts)

    model = artifacts["model"]
    probabilities = model.predict_proba(transformed)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    results = df_raw.copy()
    results["Churn_Probability"] = np.round(probabilities, 4)
    results["Prediction"] = np.where(predictions == 1, "Churn", "No Churn")
    results["Risk_Level"] = [_risk_level(p) for p in probabilities]

    return results


if __name__ == "__main__":
    # Quick manual test with one realistic sample customer
    artifacts = load_artifacts()

    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.7,
        "TotalCharges": 428.5,
    }

    result = predict_single(sample_customer, artifacts)
    print("\n--- Single Customer Prediction ---")
    print(result)