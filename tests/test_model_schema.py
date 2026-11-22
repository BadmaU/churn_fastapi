from churn_fastapi import model
from churn_fastapi.config import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN


def test_model_schema_lists_features_in_training_order(client):
    resp = client.get("/model/schema")
    assert resp.status_code == 200
    data = resp.json()
    assert [feature["name"] for feature in data["features"]] == ALL_FEATURES
    assert data["numeric_features"] == NUMERIC_FEATURES
    assert data["categorical_features"] == CATEGORICAL_FEATURES
    assert data["target_column"] == TARGET_COLUMN


def test_model_schema_reports_types_and_constraints(client):
    features = {f["name"]: f for f in client.get("/model/schema").json()["features"]}
    assert features["monthly_fee"]["type"] == "number"
    assert features["monthly_fee"]["group"] == "numeric"
    assert features["monthly_fee"]["minimum"] == 0
    assert features["support_requests"]["type"] == "integer"
    assert features["region"]["type"] == "string"
    assert features["region"]["group"] == "categorical"
    assert features["autopay_enabled"]["maximum"] == 1
    assert all(feature["required"] for feature in features.values())


def test_model_schema_available_without_trained_model(client, monkeypatch, tmp_path):
    monkeypatch.setattr(model, "MODEL_PATH", tmp_path / "absent.joblib")
    model.reset_model()
    assert client.get("/model/schema").status_code == 200
    model.reset_model()
