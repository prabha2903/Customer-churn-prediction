"""
train_model.py
--------------
Trains and compares multiple classification algorithms for churn prediction:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - XGBoost

Uses Stratified K-Fold Cross-Validation on the training set for a robust
comparison, then evaluates all models on the held-out test set, and saves
the best-performing model (by ROC AUC) to disk via joblib.
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_preprocessing import load_data, clean_data, prepare_train_test_data
from src.evaluate import evaluate_model  # we will build this in the next module


def get_models(y_train: pd.Series) -> dict:
    """
    Instantiate all candidate models with class-imbalance handling baked in.

    Parameters
    ----------
    y_train : pd.Series
        Training labels, used to compute XGBoost's scale_pos_weight.

    Returns
    -------
    dict
        Mapping of model name -> unfitted sklearn/xgboost estimator.
    """
    # scale_pos_weight = (# negative samples) / (# positive samples)
    # Tells XGBoost to penalize misclassifying the minority class (churners) more.
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=config.RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=6,              # shallow depth to limit overfitting
            random_state=config.RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=200,
            max_depth=10,
            random_state=config.RANDOM_STATE,
            n_jobs=-1                 # use all CPU cores
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=config.RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False
        ),
    }
    return models


def cross_validate_models(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Run 5-fold Stratified Cross-Validation on each model using ROC AUC
    as the scoring metric (a good single metric for imbalanced binary
    classification, since it's threshold-independent).

    Returns
    -------
    pd.DataFrame
        Mean and std of CV ROC AUC scores per model, sorted best-first.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    results = []

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
        results.append({
            "Model": name,
            "CV_ROC_AUC_Mean": scores.mean(),
            "CV_ROC_AUC_Std": scores.std()
        })
        print(f"[cross_validate_models] {name}: ROC AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

    results_df = pd.DataFrame(results).sort_values("CV_ROC_AUC_Mean", ascending=False).reset_index(drop=True)
    return results_df


def train_and_evaluate_all(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Fit each model on the FULL training set, evaluate on the held-out test set,
    and collect all metrics into a single comparison table.

    Returns
    -------
    pd.DataFrame
        Test-set metrics per model, sorted best-first by ROC AUC.
    fitted_models : dict
        The actually-fitted model objects, needed to pick and save the best one.
    """
    all_results = []
    fitted_models = {}

    for name, model in models.items():
        print(f"\n[train_and_evaluate_all] Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        all_results.append(metrics)

    results_df = pd.DataFrame(all_results).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    return results_df, fitted_models


def save_best_model(results_df: pd.DataFrame, fitted_models: dict):
    """
    Select the model with the highest ROC AUC on the test set and persist it
    to disk with joblib, so app/predict.py can load it without retraining.
    """
    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]

    joblib.dump(best_model, config.BEST_MODEL_PATH)
    print(f"\n[save_best_model] Best model: {best_model_name} "
          f"(ROC AUC: {results_df.iloc[0]['ROC_AUC']:.4f}) saved to {config.BEST_MODEL_PATH}")

    return best_model_name, best_model


if __name__ == "__main__":
    # --- Full pipeline: load -> clean -> prepare -> train -> evaluate -> save ---
    raw_df = load_data()
    cleaned_df = clean_data(raw_df)
    X_train, X_test, y_train, y_test = prepare_train_test_data(cleaned_df)

    models = get_models(y_train)

    print("\n=== Cross-Validation Results (Training Set) ===")
    cv_results = cross_validate_models(models, X_train, y_train)
    print(cv_results)

    print("\n=== Test Set Evaluation (All Models) ===")
    test_results, fitted_models = train_and_evaluate_all(models, X_train, X_test, y_train, y_test)
    print(test_results)

    best_name, best_model = save_best_model(test_results, fitted_models)