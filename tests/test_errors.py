import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churn_fastapi import dataset, main, model
from churn_fastapi.config import ALL_FEATURES, TARGET_COLUMN

ERROR_FIELDS = {"code", "message", "details"}


class TestErrorFormat:
    def test_all_errors_share_common_format(self, client):
        responses = (
            client.get("/no-such-endpoint"),
            client.post("/predict", json={"monthly_fee": "abc"}),
            client.post("/model/train", json={"model_type": "svm"}),
        )
        for response in responses:
            body = response.json()
            assert ERROR_FIELDS <= body.keys()
            assert isinstance(body["code"], str)
            assert isinstance(body["message"], str)

    def test_legacy_detail_mirrors_message(self, client):
        body = client.get("/no-such-endpoint").json()
        assert body["detail"] == body["message"]

    def test_unexpected_error_has_no_traceback(self, monkeypatch):
        def boom():
            raise Exception("секретная внутренняя ошибка")

        monkeypatch.setattr(main, "load_metrics", boom)
        with TestClient(main.app, raise_server_exceptions=False) as safe_client:
            response = safe_client.get("/metrics")

        assert response.status_code == 500
        body = response.json()
        assert body["code"] == "internal_error"
        assert "секретная внутренняя ошибка" not in response.text
        assert "Traceback" not in response.text


class TestHandledCases:
    def test_wrong_number_of_features(self, client, sample_client):
        extra = client.post("/predict", json={**sample_client, "unknown_feature": 1})
        missing = client.post(
            "/predict", json={k: v for k, v in sample_client.items() if k != "region"}
        )
        for response in (extra, missing):
            assert response.status_code == 422
            assert response.json()["code"] == "validation_error"

    def test_wrong_value_types(self, client, sample_client):
        response = client.post("/predict", json={**sample_client, "monthly_fee": "abc"})
        assert response.status_code == 422
        body = response.json()
        assert body["code"] == "validation_error"
        assert any(item["field"].endswith("monthly_fee") for item in body["details"])

    def test_empty_dataset(self, client, monkeypatch):
        empty = pd.DataFrame(columns=ALL_FEATURES + [TARGET_COLUMN])
        monkeypatch.setattr(dataset, "load_dataframe", lambda: empty)

        response = client.post("/model/train")

        assert response.status_code == 400
        assert response.json()["code"] == "empty_dataset"

    def test_no_trained_model(self, client, sample_client, monkeypatch, tmp_path):
        monkeypatch.setattr(model, "MODEL_PATH", tmp_path / "absent.joblib")
        monkeypatch.setattr(model, "_model_pipeline", None)

        response = client.post("/predict", json=sample_client)

        assert response.status_code == 400
        assert response.json()["code"] == "model_not_trained"
        assert "POST /model/train" in response.json()["message"]

    def test_invalid_feature_matrix(self, client, sample_client, monkeypatch):
        def broken(_records):
            raise ValueError("Отсутствуют обязательные признаки: ['region']")

        monkeypatch.setattr(main.preprocessing, "build_feature_matrix", broken)

        response = client.post("/predict", json=sample_client)

        assert response.status_code == 400
        assert response.json()["code"] == "invalid_features"

    def test_invalid_model_config(self, client):
        response = client.post("/model/train", json={"model_type": "svm"})

        assert response.status_code == 400
        assert response.json()["code"] == "invalid_model_config"

    @pytest.mark.parametrize(
        ("path", "code"),
        [("/no-such-endpoint", "not_found"), ("/metrics", "metrics_not_available")],
    )
    def test_http_errors(self, client, monkeypatch, path, code):
        monkeypatch.setattr(main, "load_metrics", lambda: None)

        response = client.get(path)

        assert response.json()["code"] == code
