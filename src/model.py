"""
model.py
--------
Trains, evaluates, and persists the overbought/oversold classifier.
Uses walk-forward validation and MLflow experiment tracking.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def walk_forward_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """Simple time-based train/test split — no shuffling."""
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train(feature_df: pd.DataFrame,
          n_estimators: int = 200,
          max_depth: int = 6,
          experiment_name: str = "overbought-oversold-detection"):
    """Train a Random Forest classifier with SMOTE and log to MLflow."""
    mlflow.set_experiment(experiment_name)

    train_df, test_df = walk_forward_split(feature_df)
    X_train, y_train = train_df.drop(columns=["label"]), train_df["label"]
    X_test,  y_test  = test_df.drop(columns=["label"]),  test_df["label"]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_sc, y_train)

    with mlflow.start_run():
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("smote", True)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train_res, y_train_res)

        preds = clf.predict(X_test_sc)
        acc   = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_oversold",   report.get("0", {}).get("precision", 0))
        mlflow.log_metric("precision_overbought", report.get("2", {}).get("precision", 0))
        mlflow.sklearn.log_model(clf, "random_forest_model")

        print(f"\nAccuracy: {acc:.4f}")
        print(classification_report(y_test, preds,
              target_names=["Oversold", "Neutral", "Overbought"]))

        joblib.dump(clf,    MODEL_DIR / "classifier.joblib")
        joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    return clf, scaler
