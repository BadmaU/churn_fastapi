from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from churn_fastapi import dataset
from churn_fastapi.config import METRICS_PATH, MODEL_DIR, MODEL_PATH
from churn_fastapi.preprocessing import build_pipeline, make_split

_model_pipeline: Pipeline | None = None


def _build_pipeline() -> Pipeline:
    return build_pipeline(RandomForestClassifier(n_estimators=100, random_state=42))


def train(dataset_path: Path | None = None) -> dict:
    global _model_pipeline

    if dataset_path is not None:
        df = pd.read_csv(dataset_path)
    else:
        df = dataset.load_dataframe()

    split = make_split(df)
    pipeline = _build_pipeline()
    pipeline.fit(split.X_train, split.y_train)
    _model_pipeline = pipeline

    y_pred = pipeline.predict(split.X_test)
    y_proba = pipeline.predict_proba(split.X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(split.y_test, y_pred), 4),
        "precision": round(precision_score(split.y_test, y_pred), 4),
        "recall": round(recall_score(split.y_test, y_pred), 4),
        "f1": round(f1_score(split.y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(split.y_test, y_proba), 4),
        "train_size": split.train_size,
        "test_size": split.test_size,
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
