"""
Central configuration file.
Keeps all paths and constants in one place so we never hardcode strings
across multiple modules. This is a common best-practice interviewers ask about.
"""

import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoders.joblib")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")

# Target column
TARGET_COLUMN = "Churn"

# Random state for reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2