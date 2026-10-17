from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from churn_fastapi import dataset
from churn_fastapi.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    MODEL_DIR,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from churn_fastapi.preprocessing import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    DataSplit,
    build_pipeline,
    make_split,
)
from churn_fastapi.schemas import TrainingConfigChurn

logger = logging.getLogger("uvicorn.error.churn_fastapi.model")

MODEL_NAME = "LogisticRegression"
SUPPORTED_MODEL_TYPES = ("logreg", "random_forest")

_model_pipeline: Pipeline | None = None
_loaded_from_disk = False


class InvalidDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class TrainedChurnModel:
    pipeline: Pipeline
    split: DataSplit
    metrics: dict
    model_type: str
    hyperparameters: dict


def build_classifier(config: TrainingConfigChurn | None = None):
    if config is None:
        config = TrainingConfigChurn()
    params = dict(config.hyperparameters or {})
    if config.model_type == "logreg":
        params.setdefault("max_iter", 1000)
        params.setdefault("random_state", DEFAULT_RANDOM_STATE)
        return LogisticRegression(**params)
    if config.model_type == "random_forest":
        params.setdefault("random_state", DEFAULT_RANDOM_STATE)
        return RandomForestClassifier(**params)
    raise ValueError(
        f"Неизвестный тип модели: {config.model_type}. "
        f"Доступные типы: {', '.join(SUPPORTED_MODEL_TYPES)}"
    )


def train_churn_model(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
    config: TrainingConfigChurn | None = None,
) -> TrainedChurnModel:
    if df.empty:
        raise InvalidDatasetError("Датасет пуст: нет строк для обучения")

    missing = [c for c in ALL_FEATURES + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise InvalidDatasetError(f"В датасете отсутствуют обязательные столбцы: {missing}")

    if df[TARGET_COLUMN].nunique() < 2:
        raise InvalidDatasetError("Для обучения нужны оба класса churn (0 и 1)")

    classifier = build_classifier(config)
    split = make_split(df, test_size=test_size, random_state=random_state, stratify=stratify)
    pipeline = build_pipeline(classifier)
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

    return TrainedChurnModel(
        pipeline=pipeline,
        split=split,
        metrics=metrics,
        model_type=config.model_type if config is not None else "logreg",
        hyperparameters=classifier.get_params(deep=False),
    )


def _build_metadata(result: TrainedChurnModel, df: pd.DataFrame) -> dict:
    return {
        "model": result.pipeline.named_steps["classifier"].__class__.__name__,
        "model_type": result.model_type,
        "hyperparameters": result.hyperparameters,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": result.metrics,
        "features": list(ALL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_column": TARGET_COLUMN,
        "dataset_path": str(DATASET_PATH),
        "dataset_rows": len(df),
        "sklearn_version": sklearn.__version__,
    }


def save_churn_model(pipeline: Pipeline, metadata: dict | None = None) -> Path:
    global _model_pipeline, _loaded_from_disk

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    if metadata is not None:
        MODEL_METADATA_PATH.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str)
        )

    _model_pipeline = pipeline
    _loaded_from_disk = False
    logger.info("Модель сохранена в %s", MODEL_PATH)
    return MODEL_PATH


def load_churn_model() -> Pipeline | None:
    global _model_pipeline, _loaded_from_disk

    if not MODEL_PATH.exists():
        return None

    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        logger.warning("Не удалось загрузить модель из %s: %s", MODEL_PATH, e)
        return None

    _model_pipeline = pipeline
    _loaded_from_disk = True
    logger.info("Модель загружена из %s", MODEL_PATH)
    return pipeline


def get_metadata() -> dict | None:
    if MODEL_METADATA_PATH.exists():
        try:
            return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.warning("Метаданные модели повреждены: %s", e)
    return None


def get_model_paths() -> tuple[Path, Path]:
    return MODEL_PATH, MODEL_METADATA_PATH


def is_loaded_from_disk() -> bool:
    return _loaded_from_disk


def reset_model() -> None:
    global _model_pipeline, _loaded_from_disk
    _model_pipeline = None
    _loaded_from_disk = False


def train(
    dataset_path: Path | None = None,
    config: TrainingConfigChurn | None = None,
) -> dict:
    global _model_pipeline

    if dataset_path is not None:
        df = pd.read_csv(dataset_path)
    else:
        df = dataset.load_dataframe()

    result = train_churn_model(df, config=config)
    _model_pipeline = result.pipeline
    metadata = _build_metadata(result, df)
    save_churn_model(result.pipeline, metadata)

    return metadata


def load_model() -> Pipeline | None:
    if _model_pipeline is not None:
        return _model_pipeline
    return load_churn_model()


def predict(X: pd.DataFrame) -> tuple[list[int], list[float]]:
    model = load_model()
    if model is None:
        raise RuntimeError("Модель не обучена. Сначала вызовите POST /model/train")
    preds = model.predict(X).tolist()
    probas = model.predict_proba(X)[:, 1].tolist()
    return preds, probas


def load_metrics() -> dict | None:
    metadata = get_metadata()
    if metadata is not None:
        return metadata.get("metrics")
    return None
