"""
utils.py
--------
Small shared helper functions for the Streamlit app — kept separate from
streamlit_app.py so the main app file stays focused on layout/UI logic.
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.predict import load_artifacts


@st.cache_resource
def get_artifacts():
    """
    Load model + encoders + scaler + feature names ONCE per app session
    and keep them cached in memory. @st.cache_resource is correct here
    (not @st.cache_data) because these are non-serializable ML objects,
    not plain data.
    """
    return load_artifacts()


@st.cache_data
def get_processed_dataset():
    """
    Load the cleaned dataset (used for Dashboard stats & EDA charts).
    @st.cache_data is correct here since a DataFrame is plain, hashable data.
    """
    return pd.read_csv(config.PROCESSED_DATA_PATH)


def risk_badge(risk_level: str) -> str:
    """
    Return a colored HTML badge string for a given risk level, used to
    render risk visually in the UI instead of plain text.
    """
    colors = {
        "Low": "#2E7D32",     # green
        "Medium": "#F9A825",  # amber
        "High": "#C62828",    # red
    }
    color = colors.get(risk_level, "#757575")
    return (
        f'<span style="background-color:{color}; color:white; '
        f'padding:4px 12px; border-radius:12px; font-weight:600;">'
        f'{risk_level} Risk</span>'
    )