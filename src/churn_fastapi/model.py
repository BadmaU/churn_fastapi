from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from churn_fastapi import dataset
from churn_fastapi.config import ALL_FEATURES, METRICS_PATH, MODEL_DIR, MODEL_PATH, TARGET_COLUMN
from churn_fastapi.preprocessing import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    DataSplit,
    build_pipeline,
    make_split,
)

MODEL_NAME = "LogisticRegression"

_model_pipeline: Pipeline | None = None


class InvalidDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class TrainedChurnModel:
    pipeline: Pipeline
    split: DataSplit
    metrics: dict


def build_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=DEFAULT_RANDOM_STATE)


def train_churn_model(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
) -> TrainedChurnModel:
    if df.empty:
        raise InvalidDatasetError("Датасет пуст: нет строк для обучения")

    missing = [c for c in ALL_FEATURES + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise InvalidDatasetError(f"В датасете отсутствуют обязательные столбцы: {missing}")

    if df[TARGET_COLUMN].nunique() < 2:
        raise InvalidDatasetError("Для обучения нужны оба класса churn (0 и 1)")

    split = make_split(df, test_size=test_size, random_state=random_state, stratify=stratify)
    pipeline = build_pipeline(build_classifier())
    pipeline.fit(split.X_train, split.y_train)

    y_pred = pipeline.predict(split.X_test)
    y_proba = pipeline.predict_proba(split.X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(split.y_test, y_pred), 4),
        "precision": round(precision_score(split.y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(split.y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(split.y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(split.y_test, y_proba), 4),
        "train_size": split.train_size,
        "test_size": split.test_size,
    }

    return TrainedChurnModel(pipeline=pipeline, split=split, metrics=metrics)


def train(dataset_path: Path | None = None) -> dict:
    global _model_pipeline

    if dataset_path is not None:
        df = pd.read_csv(dataset_path)
    else:
        df = dataset.load_dataframe()

    result = train_churn_model(df)
    _model_pipeline = result.pipeline

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(result.metrics, indent=2))

    return result.metrics


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
        raise RuntimeError("Модель не обучена. Сначала вызовите POST /model/train")
    preds = model.predict(X).tolist()
    probas = model.predict_proba(X)[:, 1].tolist()
    return preds, probas


def load_metrics() -> dict | None:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None
