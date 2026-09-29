from __future__ import annotations

import pandas as pd

from churn_fastapi.config import ALL_FEATURES, DATASET_PATH, TARGET_COLUMN
from churn_fastapi.schemas import DatasetRowChurn

_cache: pd.DataFrame | None = None
_cache_stamp: tuple[float, int] | None = None


class DatasetNotFoundError(FileNotFoundError):
    pass


def _current_stamp() -> tuple[float, int]:
    stat = DATASET_PATH.stat()
    return stat.st_mtime, stat.st_size


def load_dataframe() -> pd.DataFrame:
    global _cache, _cache_stamp

    if not DATASET_PATH.exists():
        raise DatasetNotFoundError(f"Файл датасета не найден: {DATASET_PATH}")

    stamp = _current_stamp()
    if _cache is None or stamp != _cache_stamp:
        df = pd.read_csv(DATASET_PATH)
        missing = [c for c in ALL_FEATURES + [TARGET_COLUMN] if c not in df.columns]
        if missing:
            raise ValueError(f"В датасете отсутствуют обязательные столбцы: {missing}")
        _cache = df
        _cache_stamp = stamp

    return _cache


def get_rows(limit: int | None = None) -> list[DatasetRowChurn]:
    df = load_dataframe()
    if limit is not None:
        df = df.head(limit)
    return [DatasetRowChurn(**row) for row in df.to_dict(orient="records")]


def count_rows() -> int:
    return len(load_dataframe())


def get_features() -> list[str]:
    return list(ALL_FEATURES)


def get_churn_distribution() -> dict[str, int]:
    df = load_dataframe()
    return {str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().sort_index().items()}


def get_churn_ratio() -> float:
    df = load_dataframe()
    if df.empty:
        return 0.0
    return round(float(df[TARGET_COLUMN].mean()), 4)


def clear_cache() -> None:
    global _cache, _cache_stamp
    _cache = None
    _cache_stamp = None
