from churn_fastapi import dataset
from churn_fastapi.config import ALL_FEATURES, TARGET_COLUMN
from churn_fastapi.schemas import DatasetRowChurn


def test_dataset_info(client):
    resp = client.get("/dataset/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["rows"] == 2000
    assert data["columns"] == 10
    assert data["features"] == ALL_FEATURES
    assert data["target_column"] == TARGET_COLUMN
    assert sum(data["churn_distribution"].values()) == 2000
    assert set(data["churn_distribution"]) == {"0", "1"}
    assert 0.0 < data["churn_ratio"] < 1.0


def test_dataset_legacy_endpoint_still_works(client):
    resp = client.get("/dataset")
    assert resp.status_code == 200
    data = resp.json()
    assert data["rows"] == 2000
    assert "churn" in data["columns"]


def test_dataset_preview_default_limit(client):
    resp = client.get("/dataset/preview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_rows"] == 2000
    assert data["limit"] == 10
    assert data["returned"] == 10
    assert len(data["rows"]) == 10
    first = data["rows"][0]
    assert set(first) == set(ALL_FEATURES) | {TARGET_COLUMN}
    assert first["monthly_fee"] == 9.99
    assert first["region"] == "america"


def test_dataset_preview_custom_limit(client):
    resp = client.get("/dataset/preview", params={"limit": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["returned"] == 3
    assert len(data["rows"]) == 3


def test_dataset_preview_limit_larger_than_dataset(client):
    resp = client.get("/dataset/preview", params={"limit": 5000})
    assert resp.status_code == 200
    data = resp.json()
    assert data["returned"] == 2000


def test_dataset_preview_invalid_limit(client):
    assert client.get("/dataset/preview", params={"limit": 0}).status_code == 422
    assert client.get("/dataset/preview", params={"limit": -5}).status_code == 422
    assert client.get("/dataset/preview", params={"limit": 10001}).status_code == 422


def test_dataset_rows_are_valid_models():
    rows = dataset.get_rows(limit=5)
    assert len(rows) == 5
    assert all(isinstance(r, DatasetRowChurn) for r in rows)
    assert all(r.churn in (0, 1) for r in rows)
    assert all(r.autopay_enabled in (0, 1) for r in rows)
    assert all(r.account_age_months >= 1 for r in rows)


def test_dataset_module_helpers():
    assert dataset.count_rows() == 2000
    assert dataset.get_features() == ALL_FEATURES
    distribution = dataset.get_churn_distribution()
    assert sum(distribution.values()) == 2000
    assert 0.0 < dataset.get_churn_ratio() < 1.0


def test_endpoints_return_404_when_dataset_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(dataset, "DATASET_PATH", tmp_path / "missing.csv")
    dataset.clear_cache()
    try:
        assert client.get("/dataset/info").status_code == 404
        assert client.get("/dataset/preview").status_code == 404
        assert client.get("/dataset").status_code == 404
    finally:
        dataset.clear_cache()


def test_endpoints_return_422_on_broken_dataset(client, monkeypatch, tmp_path):
    broken = tmp_path / "broken.csv"
    broken.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    monkeypatch.setattr(dataset, "DATASET_PATH", broken)
    dataset.clear_cache()
    try:
        resp = client.get("/dataset/info")
        assert resp.status_code == 422
        assert "monthly_fee" in resp.json()["detail"]
    finally:
        dataset.clear_cache()
