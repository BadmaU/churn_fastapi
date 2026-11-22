from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from churn_fastapi.core.config import HISTORY_PATH

logger = logging.getLogger("uvicorn.error.churn_fastapi.history")

MAX_RECORDS = 100


def load_history(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or HISTORY_PATH
    if not target.exists():
        return []
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Не удалось прочитать историю обучений %s: %s", target, e)
        return []
    if not isinstance(data, list):
        logger.warning("История обучений %s имеет неожиданный формат", target)
        return []
    return [record for record in data if isinstance(record, dict)]


def append_record(record: dict[str, Any], path: Path | None = None) -> None:
    target = path or HISTORY_PATH
    records = load_history(target)
    records.append(record)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(records[-MAX_RECORDS:], indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def filter_history(
    records: list[dict[str, Any]], model_type: str | None = None
) -> list[dict[str, Any]]:
    if model_type is None:
        return records
    return [record for record in records if record.get("model_type") == model_type]


def best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None

    def score(record: dict[str, Any]) -> tuple[float, float]:
        metrics = record.get("metrics") or {}
        return (
            float(metrics.get("f1") or 0),
            float(metrics.get("accuracy") or 0),
        )

    return max(records, key=score)


def get_training_metrics(
    limit: int = 10,
    model_type: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    records = filter_history(load_history(path), model_type)
    return {
        "latest": records[-1] if records else None,
        "history": records[-limit:],
        "best": best_record(records),
        "total": len(records),
    }
