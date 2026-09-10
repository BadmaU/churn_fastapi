import pytest
from fastapi.testclient import TestClient

from churn_fastapi.main import app


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
