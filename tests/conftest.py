import random

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churn_fastapi import dataset, history
from churn_fastapi.config import ALL_FEATURES, TARGET_COLUMN
from churn_fastapi.main import app


@pytest.fixture(autouse=True)
def isolated_training_history(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_PATH", tmp_path / "training_history.json")


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_client():
    return {
        "monthly_fee": 19.99,
        "usage_hours": 15.0,
        "support_requests": 2,
        "account_age_months": 6,
        "failed_payments": 0,
        "region": "europe",
        "device_type": "desktop",
        "payment_method": "card",
        "autopay_enabled": 1,
    }


@pytest.fixture
def synthetic_df():
    rng = random.Random(42)
    rows = []
    for index in range(60):
        churn = index % 2
        rows.append(
            {
                "monthly_fee": round(rng.uniform(10, 100) + churn * 5, 2),
                "usage_hours": round(rng.uniform(1, 100) - churn * 2, 2),
                "support_requests": rng.randint(0, 5),
                "account_age_months": rng.randint(1, 48),
                "failed_payments": rng.randint(0, 2),
                "region": rng.choice(["europe", "asia", "america", "africa"]),
                "device_type": rng.choice(["mobile", "desktop", "tablet"]),
                "payment_method": rng.choice(["card", "paypal", "crypto"]),
                "autopay_enabled": rng.randint(0, 1),
                TARGET_COLUMN: churn,
            }
        )
    return pd.DataFrame(rows, columns=ALL_FEATURES + [TARGET_COLUMN])


@pytest.fixture
def synthetic_csv(tmp_path, monkeypatch, synthetic_df):
    path = tmp_path / "churn_dataset.csv"
    synthetic_df.to_csv(path, index=False)
    monkeypatch.setattr(dataset, "DATASET_PATH", path)
    dataset.clear_cache()
    yield path
    dataset.clear_cache()
