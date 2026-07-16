"""
streamlit_app.py
----------------
Main Streamlit application for the Customer Churn Prediction System.

Structure:
    - Page config & global styling
    - Cached loading of model artifacts and dataset
    - Tabbed layout: Dashboard | Single Prediction | Bulk Prediction | Feature Importance
"""

import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from app.utils import get_artifacts, get_processed_dataset
from src.eda import (
    plot_target_distribution,
    plot_categorical_vs_churn,
    plot_numerical_distribution,
    plot_correlation_heatmap,
)

# --- Page configuration (MUST be the first Streamlit call) ---
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load cached artifacts and data once ---
artifacts = get_artifacts()
df = get_processed_dataset()

# --- Sidebar branding ---
with st.sidebar:
    st.title("📉 Churn Predictor")
    st.markdown("---")
    st.markdown(
        "**About this app**\n\n"
        "An end-to-end ML system that predicts telecom customer churn "
        "using Logistic Regression, Decision Tree, Random Forest, and "
        "XGBoost — with a full analytics dashboard and prediction tools."
    )
    st.markdown("---")
    st.markdown(f"**Model in use:** `{type(artifacts['model']).__name__}`")
    st.markdown(f"**Total records:** {df.shape[0]:,}")

# --- Main title ---
st.title("Customer Churn Prediction System")
st.caption("End-to-end ML pipeline: EDA → Feature Engineering → Model Comparison → Deployment")

# --- Tabbed navigation ---
tab_dashboard, tab_single, tab_bulk, tab_importance = st.tabs(
    ["📊 Dashboard", "🧑 Single Prediction", "📁 Bulk Prediction", "🔍 Feature Importance"]
)

# =========================================================
# TAB 1: DASHBOARD
# =========================================================
with tab_dashboard:
    st.subheader("Dataset Overview")

    # --- KPI row ---
    col1, col2, col3, col4 = st.columns(4)

    total_customers = df.shape[0]
    churn_rate = df[config.TARGET_COLUMN].mean() * 100
    avg_tenure = df["tenure"].mean()
    avg_monthly_charge = df["MonthlyCharges"].mean()

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churn Rate", f"{churn_rate:.1f}%")
    col3.metric("Avg. Tenure", f"{avg_tenure:.1f} months")
    col4.metric("Avg. Monthly Charge", f"${avg_monthly_charge:.2f}")

    st.markdown("---")

    # --- Row 1: Target distribution + a categorical driver ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_target_distribution(df), use_container_width=True)
    with col_b:
        st.plotly_chart(plot_categorical_vs_churn(df, "Contract"), use_container_width=True)

    # --- Row 2: pick any categorical column interactively ---
    st.markdown("#### Explore Churn Rate by Category")
    categorical_options = [
        "InternetService", "PaymentMethod", "PaperlessBilling",
        "SeniorCitizen", "Partner", "Dependents", "TechSupport", "OnlineSecurity"
    ]
    selected_cat = st.selectbox("Choose a feature to explore:", categorical_options)
    st.plotly_chart(plot_categorical_vs_churn(df, selected_cat), use_container_width=True)

    # --- Row 3: numeric distribution + correlation heatmap ---
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(plot_numerical_distribution(df, "tenure"), use_container_width=True)
    with col_d:
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

# =========================================================
# TAB 2: SINGLE PREDICTION  (placeholder — built in Module 7b)
# =========================================================
with tab_single:
    st.info("🚧 Coming in the next step: single customer prediction form.")

# =========================================================
# TAB 3: BULK PREDICTION  (placeholder — built in Module 7c)
# =========================================================
with tab_bulk:
    st.info("🚧 Coming in the next step: bulk CSV prediction.")

# =========================================================
# TAB 4: FEATURE IMPORTANCE  (placeholder — built in Module 7d)
# =========================================================
with tab_importance:
    st.info("🚧 Coming in the next step: feature importance visualization.")