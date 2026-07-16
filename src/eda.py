"""
eda.py
------
Reusable exploratory data analysis functions.
Each function returns a Plotly figure object so it can be used both:
    1. Standalone (python src/eda.py) for exploration during development
    2. Inside the Streamlit dashboard (Module: app/streamlit_app.py) for reuse

Keeping EDA as functions (not notebook cells) avoids duplicating plotting
code between our exploration phase and the final app.
"""

import pandas as pd
import plotly.express as px
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def plot_target_distribution(df: pd.DataFrame):
    """
    Pie chart of Churn vs Non-Churn to visualize class imbalance.
    """
    churn_counts = df[config.TARGET_COLUMN].map({1: "Churned", 0: "Retained"}).value_counts()
    fig = px.pie(
        names=churn_counts.index,
        values=churn_counts.values,
        title="Customer Churn Distribution",
        color=churn_counts.index,
        color_discrete_map={"Retained": "#2E86AB", "Churned": "#E63946"},
        hole=0.4,
    )
    return fig


def plot_categorical_vs_churn(df: pd.DataFrame, column: str):
    """
    Grouped bar chart showing churn rate (%) across categories of a
    given categorical column (e.g., Contract, InternetService, PaymentMethod).
    """
    grouped = (
        df.groupby(column)[config.TARGET_COLUMN]
        .mean()
        .mul(100)
        .round(2)
        .reset_index()
        .rename(columns={config.TARGET_COLUMN: "ChurnRate"})
    )
    fig = px.bar(
        grouped,
        x=column,
        y="ChurnRate",
        title=f"Churn Rate (%) by {column}",
        text="ChurnRate",
        color="ChurnRate",
        color_continuous_scale="Reds",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(yaxis_title="Churn Rate (%)")
    return fig


def plot_numerical_distribution(df: pd.DataFrame, column: str):
    """
    Overlaid histogram comparing the distribution of a numeric column
    (e.g., tenure, MonthlyCharges) between churned and retained customers.
    """
    plot_df = df.copy()
    plot_df["ChurnLabel"] = plot_df[config.TARGET_COLUMN].map({1: "Churned", 0: "Retained"})

    fig = px.histogram(
        plot_df,
        x=column,
        color="ChurnLabel",
        barmode="overlay",
        nbins=40,
        title=f"Distribution of {column} by Churn Status",
        color_discrete_map={"Retained": "#2E86AB", "Churned": "#E63946"},
        opacity=0.7,
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Correlation heatmap for numeric columns to detect multicollinearity
    (e.g., tenure, MonthlyCharges, TotalCharges are naturally related).
    """
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap (Numeric Features)",
        zmin=-1,
        zmax=1,
    )
    return fig


if __name__ == "__main__":
    df = pd.read_csv(config.PROCESSED_DATA_PATH)

    # Quick sanity check: show figures one by one when run standalone
    fig1 = plot_target_distribution(df)
    fig1.show()

    fig2 = plot_categorical_vs_churn(df, "InternetService")
    fig2.show()

    fig3 = plot_numerical_distribution(df, "tenure")
    fig3.show()

    fig4 = plot_correlation_heatmap(df)
    fig4.show()