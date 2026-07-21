"""
streamlit_app.py
----------------
Main Streamlit application for the Customer Churn Prediction System.

Structure:
    - Page config & global styling
    - Cached loading of model artifacts and dataset
    - Tabbed layout: Dashboard | Single Prediction | Bulk Prediction | Feature Importance
"""
"""
streamlit_app.py
----------------
Main Streamlit application for the Customer Churn Prediction System.
"""

import streamlit as st
import sys
import os

# This MUST run before any "from src..." or "from app..." import below
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict_bulk
import pandas as pd
import plotly.express as px
import config
from app.utils import get_artifacts, get_processed_dataset, risk_badge
from src.eda import (
    plot_target_distribution,
    plot_categorical_vs_churn,
    plot_numerical_distribution,
    plot_correlation_heatmap,
)
from src.predict import predict_single

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
    st.subheader("Predict Churn for a Single Customer")
    st.caption("Fill in customer details below to get a real-time churn prediction.")

    with st.form("single_prediction_form"):

        # --- Section 1: Demographics ---
        st.markdown("#### 👤 Demographics")
        d1, d2, d3, d4 = st.columns(4)
        gender = d1.selectbox("Gender", ["Female", "Male"])
        senior_citizen = d2.selectbox("Senior Citizen", ["No", "Yes"])
        partner = d3.selectbox("Has Partner", ["Yes", "No"])
        dependents = d4.selectbox("Has Dependents", ["Yes", "No"])

        # --- Section 2: Account Info ---
        st.markdown("#### 💳 Account Information")
        a1, a2, a3 = st.columns(3)
        tenure = a1.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        contract = a2.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = a3.selectbox("Paperless Billing", ["Yes", "No"])

        a4, a5 = st.columns(2)
        payment_method = a4.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = a5.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)

        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, max_value=10000.0,
            value=float(monthly_charges) * float(tenure), step=1.0,
            help="Defaults to Monthly Charges × Tenure; adjust if known exactly."
        )

        # --- Section 3: Services ---
        st.markdown("#### 📡 Services")
        s1, s2, s3 = st.columns(3)
        phone_service = s1.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = s2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = s3.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

        s4, s5, s6 = st.columns(3)
        online_security = s4.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = s5.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = s6.selectbox("Device Protection", ["No", "Yes", "No internet service"])

        s7, s8, s9 = st.columns(3)
        tech_support = s7.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = s8.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = s9.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        customer_data = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        result = predict_single(customer_data, artifacts)

        st.markdown("---")
        st.markdown("### Prediction Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            label = "Will Churn" if result["prediction"] == 1 else "Will Stay"
            st.metric("Prediction", label)

        with r2:
            st.metric("Churn Probability", f"{result['churn_probability'] * 100:.1f}%")

        with r3:
            st.markdown("**Risk Level**")
            st.markdown(risk_badge(result["risk_level"]), unsafe_allow_html=True)

        st.progress(result["churn_probability"])

        # --- Plain-English recommendation based on risk ---
        if result["risk_level"] == "High":
            st.error(
                "⚠️ **High risk of churn.** Recommend immediate retention outreach — "
                "consider a loyalty discount, contract upgrade incentive, or personal check-in call."
            )
        elif result["risk_level"] == "Medium":
            st.warning(
                "🟡 **Moderate churn risk.** Consider proactive engagement — "
                "highlight underused services or offer a small loyalty perk."
            )
        else:
            st.success(
                "✅ **Low churn risk.** Customer appears stable — standard engagement is sufficient."
            )

# =========================================================
# TAB 3: BULK PREDICTION  (placeholder — built in Module 7c)
# =========================================================
with tab_bulk:
    st.subheader("Bulk Churn Prediction via CSV Upload")
    st.caption(
        "Upload a CSV with the same columns as the Telco dataset "
        "(customerID and Churn columns are optional and will be ignored)."
    )

    # --- Sample template download ---
    with st.expander("📄 Need a template? Download a sample CSV first"):
        sample_template = df.drop(columns=[config.TARGET_COLUMN]).head(5)
        st.dataframe(sample_template, use_container_width=True)
        st.download_button(
            label="⬇️ Download Sample Template CSV",
            data=sample_template.to_csv(index=False).encode("utf-8"),
            file_name="churn_prediction_template.csv",
            mime="text/csv",
        )

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Could not read the uploaded file. Error: {e}")
            raw_df = None

        if raw_df is not None:
            st.markdown("#### 👀 Preview of Uploaded Data")
            st.dataframe(raw_df.head(), use_container_width=True)

            # --- Validate required columns exist ---
            required_cols = set(df.drop(columns=[config.TARGET_COLUMN]).columns) - {"customerID"}
            uploaded_cols = set(raw_df.columns)
            missing_cols = required_cols - uploaded_cols

            if missing_cols:
                st.error(
                    f"❌ Uploaded CSV is missing required columns: "
                    f"{', '.join(sorted(missing_cols))}. "
                    f"Please use the sample template above as a reference."
                )
            else:
                if st.button("🚀 Run Bulk Prediction", use_container_width=True):
                    with st.spinner("Scoring customers..."):
                        results = predict_bulk(raw_df, artifacts)

                    st.success(f"✅ Successfully scored {results.shape[0]} customers!")

                    # --- Summary metrics ---
                    st.markdown("#### 📊 Batch Summary")
                    m1, m2, m3, m4 = st.columns(4)

                    total = results.shape[0]
                    high_risk = (results["Risk_Level"] == "High").sum()
                    medium_risk = (results["Risk_Level"] == "Medium").sum()
                    low_risk = (results["Risk_Level"] == "Low").sum()

                    m1.metric("Total Customers", f"{total:,}")
                    m2.metric("🔴 High Risk", f"{high_risk:,}", f"{high_risk/total*100:.1f}%")
                    m3.metric("🟡 Medium Risk", f"{medium_risk:,}", f"{medium_risk/total*100:.1f}%")
                    m4.metric("🟢 Low Risk", f"{low_risk:,}", f"{low_risk/total*100:.1f}%")

                    # --- Risk distribution chart ---
                    risk_counts = results["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"])
                    fig = px.bar(
                        x=risk_counts.index,
                        y=risk_counts.values,
                        color=risk_counts.index,
                        color_discrete_map={"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"},
                        title="Risk Level Distribution (Uploaded Batch)",
                        labels={"x": "Risk Level", "y": "Number of Customers"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # --- Full results table ---
                    st.markdown("#### 📋 Full Prediction Results")
                    st.dataframe(results, use_container_width=True)

                    # --- Download button ---
                    st.download_button(
                        label="⬇️ Download Results as CSV",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name="churn_predictions_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

# =========================================================
# TAB 4: FEATURE IMPORTANCE  (placeholder — built in Module 7d)
# =========================================================
with tab_importance:
    st.info("🚧 Coming in the next step: feature importance visualization.")