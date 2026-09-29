import json

import pytest
from sklearn.pipeline import Pipeline

from churn_fastapi import model
from churn_fastapi.config import ALL_FEATURES, MODEL_METADATA_PATH, MODEL_PATH


@pytest.fixture(autouse=True)
def clean_model_state():
    model.reset_model()
    yield
    model.reset_model()


@pytest.fixture
def trained(client):
    assert client.post("/model/train").status_code == 200
    return client


class TestSaveAndLoad:
    def test_training_writes_model_file(self, trained):
        assert MODEL_PATH.exists()
        assert MODEL_PATH.name == "churn_model.joblib"
        assert MODEL_PATH.stat().st_size > 0

    def test_training_writes_metadata(self, trained):
        assert MODEL_METADATA_PATH.exists()
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        assert metadata["model"] == "LogisticRegression"
        assert "trained_at" in metadata
        assert metadata["dataset_rows"] == 2000
        assert metadata["features"] == ALL_FEATURES
        assert metadata["target_column"] == "churn"
        assert "accuracy" in metadata["metrics"]
        assert "f1" in metadata["metrics"]

    def test_save_and_load_roundtrip(self, trained):
        model.reset_model()
        assert model.load_churn_model() is None or True
        model.reset_model()
        pipeline = model.load_churn_model()
        assert isinstance(pipeline, Pipeline)
        assert model.is_loaded_from_disk() is True

    def test_load_returns_none_without_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(model, "MODEL_PATH", tmp_path / "nope.joblib")
        assert model.load_churn_model() is None
        assert model.is_loaded_from_disk() is False

    def test_corrupt_model_file_does_not_crash(self, monkeypatch, tmp_path, caplog):
        corrupt = tmp_path / "corrupt.joblib"
        corrupt.write_bytes(b"not a joblib file at all")
        monkeypatch.setattr(model, "MODEL_PATH", corrupt)
        with caplog.at_level("WARNING"):
            assert model.load_churn_model() is None
        assert model.is_loaded_from_disk() is False
        assert "Не удалось загрузить модель" in caplog.text

    def test_corrupt_metadata_is_tolerated(self, monkeypatch, tmp_path):
        bad = tmp_path / "model_metadata.json"
        bad.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(model, "MODEL_METADATA_PATH", bad)
        assert model.get_metadata() is None

    def test_load_metrics_from_metadata(self, trained):
        metrics = model.load_metrics()
        assert metrics is not None
        assert "accuracy" in metrics
        assert "roc_auc" in metrics

    def test_loaded_pipeline_predicts(self, trained, sample_client):
        import pandas as pd

        model.reset_model()
        assert model.load_model() is not None
        df = pd.DataFrame([sample_client])[ALL_FEATURES]
        preds, probas = model.predict(df)
        assert preds[0] in (0, 1)
        assert 0.0 <= probas[0] <= 1.0


class TestModelStatusEndpoint:
    def test_status_before_training(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(model, "MODEL_PATH", tmp_path / "absent.joblib")
        monkeypatch.setattr(model, "MODEL_METADATA_PATH", tmp_path / "absent.json")
        model.reset_model()
        resp = client.get("/model/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trained"] is False
        assert data["trained_at"] is None
        assert data["metrics"] is None
        assert data["model"] is None
        assert data["model_file_exists"] is False
        assert data["loaded_from_disk"] is False
        assert data["features"] == ALL_FEATURES

    def test_status_after_training(self, trained):
        data = trained.get("/model/status").json()
        assert data["trained"] is True
        assert data["model"] == "LogisticRegression"
        assert data["trained_at"] is not None
        assert data["age_seconds"] >= 0
        assert data["model_file_exists"] is True
        assert data["loaded_from_disk"] is False
        assert data["dataset_rows"] == 2000
        assert data["sklearn_version"] is not None

    def test_status_metrics_match_training_response(self, trained):
        train_data = trained.post("/model/train").json()
        status_data = trained.get("/model/status").json()
        assert status_data["metrics"]["accuracy"] == train_data["accuracy"]
        assert status_data["metrics"]["f1"] == train_data["f1"]

    def test_status_reports_load_from_disk(self, trained):
        model.reset_model()
        data = trained.get("/model/status").json()
        assert data["trained"] is True
        assert data["loaded_from_disk"] is True

    def test_status_after_surviving_restart(self, trained, sample_client):
        before = trained.post("/predict", json=sample_client).json()

        model.reset_model()
        with trained:
            trained.get("/model/status")
            assert model.load_churn_model() is not None
            after = trained.post("/predict", json=sample_client).json()

        assert before == after


class TestStartupLoad:
    def test_lifespan_loads_model_from_disk(self, trained, sample_client):
        from fastapi.testclient import TestClient

        from churn_fastapi.main import app

        model.reset_model()
        with TestClient(app) as fresh_client:
            assert fresh_client.get("/model/status").json()["loaded_from_disk"] is True
            assert fresh_client.get("/health").json()["model_loaded"] is True
            assert fresh_client.post("/predict", json=sample_client).status_code == 200

    def test_lifespan_without_model_file(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient

        from churn_fastapi.main import app

        monkeypatch.setattr(model, "MODEL_PATH", tmp_path / "absent.joblib")
        monkeypatch.setattr(model, "MODEL_METADATA_PATH", tmp_path / "absent.json")
        model.reset_model()
        with TestClient(app) as fresh_client:
            data = fresh_client.get("/model/status").json()
            assert data["trained"] is False
            assert fresh_client.get("/health").json()["model_loaded"] is False
