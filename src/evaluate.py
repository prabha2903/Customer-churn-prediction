"""
evaluate.py
-----------
Model evaluation utilities: computes classification metrics and produces
reusable Plotly visualizations (confusion matrix, ROC curve) that are used
both during training/comparison (train_model.py) and inside the Streamlit
dashboard for the final deployed model.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Compute standard classification metrics for a fitted model on held-out
    test data.

    Parameters
    ----------
    model : fitted sklearn/xgboost estimator
        Must support .predict() and .predict_proba()
    X_test, y_test : test features and true labels
    model_name : str
        Label used in the returned dict, for building comparison tables.

    Returns
    -------
    dict
        {Model, Accuracy, Precision, Recall, F1, ROC_AUC}
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of class "1" (churn)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    }

    print(
        f"[evaluate_model] {model_name} -> "
        f"Acc: {metrics['Accuracy']:.4f} | "
        f"Prec: {metrics['Precision']:.4f} | "
        f"Recall: {metrics['Recall']:.4f} | "
        f"F1: {metrics['F1']:.4f} | "
        f"ROC AUC: {metrics['ROC_AUC']:.4f}"
    )

    return metrics


def plot_confusion_matrix(model, X_test, y_test, model_name: str = "Model"):
    """
    Build an annotated confusion matrix heatmap as a Plotly figure.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    labels = ["Retained (0)", "Churned (1)"]

    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        title=f"Confusion Matrix — {model_name}",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def plot_roc_curve(model, X_test, y_test, model_name: str = "Model"):
    """
    Plot the ROC curve (True Positive Rate vs False Positive Rate across
    all thresholds) with the AUC score annotated.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"{model_name} (AUC = {auc_score:.3f})",
        line=dict(width=3)
    ))
    # Diagonal reference line = a random/no-skill classifier
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Classifier",
        line=dict(dash="dash", color="gray")
    ))

    fig.update_layout(
        title=f"ROC Curve — {model_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.55, y=0.05),
    )
    return fig


def plot_model_comparison(results_df: pd.DataFrame):
    """
    Grouped bar chart comparing all models across all metrics side-by-side.
    Expects results_df with columns: Model, Accuracy, Precision, Recall, F1, ROC_AUC
    """
    melted = results_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"],
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Model Comparison Across All Metrics",
        text_auto=".3f",
    )
    fig.update_layout(yaxis_range=[0, 1.05])
    return fig