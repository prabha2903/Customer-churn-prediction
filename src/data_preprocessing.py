"""
data_preprocessing.py
----------------------
Handles loading, cleaning, and preparing the Telco Customer Churn dataset.
This module is split into stages:
    1. load_data()      -> read raw CSV
    2. clean_data()      -> fix dtypes, handle missing values, drop junk columns
    3. encode_features() -> (Module 2) encode categoricals, scale numerics
"""

import pandas as pd
import numpy as np
import sys
import os

# Allow importing config.py from project root when this file is run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn CSV into a DataFrame.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified dataset.
    """
    df = pd.read_csv(path)
    print(f"[load_data] Loaded dataset with shape: {df.shape}")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Print a quick diagnostic summary of the dataset.
    Used during development to understand data quality issues.
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

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_data()

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering.
    """
    df = df.copy()  # never mutate the caller's original dataframe

    # --- Fix TotalCharges ---
    # It's stored as object dtype because some rows have " " (blank space)
    # instead of a number, typically for customers with tenure == 0
    # (they haven't been billed yet).
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # For tenure == 0 customers, TotalCharges should logically be 0
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


if __name__ == "__main__":
    # Allows running this file directly for a quick sanity check:
    # python src/data_preprocessing.py
    raw_df = load_data()
    inspect_data(raw_df)
    cleaned_df = clean_data(raw_df)

    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    cleaned_df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"[main] Saved cleaned data to {config.PROCESSED_DATA_PATH}")