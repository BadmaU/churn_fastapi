from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_fastapi.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def test_size(self) -> int:
        return len(self.X_test)


def get_column_groups() -> tuple[list[str], list[str]]:
    numeric = list(NUMERIC_FEATURES)
    categorical = list(CATEGORICAL_FEATURES)
    overlap = set(numeric) & set(categorical)
    if overlap:
        raise ValueError(f"Столбцы попали в обе группы: {sorted(overlap)}")
    missing = set(ALL_FEATURES) - set(numeric) - set(categorical)
    if missing:
        raise ValueError(f"Признаки не отнесены ни к одной группе: {sorted(missing)}")
    return numeric, categorical


def describe_missing(df: pd.DataFrame) -> dict[str, int]:
    return {col: int(n) for col, n in df[ALL_FEATURES].isna().sum().items() if n}


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    numeric, categorical = get_column_groups()
    features = numeric + categorical
    X = df[features].copy()
    y = df[TARGET_COLUMN].copy()
    y.name = TARGET_COLUMN
    return X, y


def make_split(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
) -> DataSplit:
    if not 0 < test_size < 1:
        raise ValueError("test_size должен быть в интервале (0, 1)")
    X, y = split_xy(df)
    stratify_target = y if stratify else None
    if stratify and y.nunique() < 2:
        raise ValueError("Нельзя стратифицировать: в churn всего один класс")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )
    return DataSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def build_preprocessor() -> ColumnTransformer:
    numeric, categorical = get_column_groups()
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def build_pipeline(classifier) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def churn_distribution(y: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in y.value_counts().sort_index().items()}


def churn_ratio(y: pd.Series) -> float:
    if len(y) == 0:
        return 0.0
    return round(float(y.mean()), 4)


def summarize_split(
    split: DataSplit,
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> dict:
    full_y = df[TARGET_COLUMN]
    train_dist = churn_distribution(split.y_train)
    test_dist = churn_distribution(split.y_test)

    return {
        "test_size": test_size,
        "random_state": random_state,
        "stratify": stratify,
        "total_rows": len(df),
        "train_size": split.train_size,
        "test_rows": split.test_size,
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "features": list(ALL_FEATURES),
        "target_column": TARGET_COLUMN,
        "missing_values": describe_missing(df),
        "missing_values_total": sum(describe_missing(df).values()),
        "churn_distribution": {
            "full": churn_distribution(full_y),
            "train": train_dist,
            "test": test_dist,
        },
        "churn_ratio": {
            "full": churn_ratio(full_y),
            "train": churn_ratio(split.y_train),
            "test": churn_ratio(split.y_test),
        },
        "stratified_consistent": (
            train_dist.keys() == test_dist.keys()
            and abs(
                churn_ratio(split.y_train) - churn_ratio(split.y_test)
            )
            <= 0.02
        ),
    }
