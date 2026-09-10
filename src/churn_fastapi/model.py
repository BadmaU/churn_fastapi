from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_fastapi.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

_model_pipeline: Pipeline | None = None


def _build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )


def train(dataset_path: Path | None = None) -> dict:
    global _model_pipeline

    path = dataset_path or DATASET_PATH
    df = pd.read_csv(path)

    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)
    _model_pipeline = pipeline

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    return metrics


def load_model() -> Pipeline | None:
    global _model_pipeline
    if _model_pipeline is not None:
        return _model_pipeline
    if MODEL_PATH.exists():
        _model_pipeline = joblib.load(MODEL_PATH)
        return _model_pipeline
    return None


def predict(X: pd.DataFrame) -> tuple[list[int], list[float]]:
    model = load_model()
    if model is None:
        raise RuntimeError("Модель не обучена. Сначала вызовите POST /train")
    preds = model.predict(X).tolist()
    probas = model.predict_proba(X)[:, 1].tolist()
    return preds, probas


def load_metrics() -> dict | None:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None
