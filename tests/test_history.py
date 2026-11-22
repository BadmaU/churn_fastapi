import json

import pytest

from churn_fastapi.ml import history, model


@pytest.fixture(autouse=True)
def isolated_model_files(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "MODEL_PATH", tmp_path / "churn_model.joblib")
    monkeypatch.setattr(model, "MODEL_METADATA_PATH", tmp_path / "model_metadata.json")
    model.reset_model()
    yield
    model.reset_model()


class TestHistoryStorage:
    def test_record_saved_after_training(self, client):
        response = client.post(
            "/model/train",
            json={"model_type": "random_forest", "hyperparameters": {"n_estimators": 10}},
        )

        assert response.status_code == 200
        records = history.load_history()
        assert len(records) == 1
        record = records[0]
        assert record["model_type"] == "random_forest"
        assert record["model"] == "RandomForestClassifier"
        assert record["hyperparameters"]["n_estimators"] == 10
        assert record["trained_at"]
        assert {"accuracy", "f1", "roc_auc"} <= record["metrics"].keys()

    def test_records_append_in_order(self, client):
        for model_type in ("logreg", "random_forest", "logreg"):
            client.post("/model/train", json={"model_type": model_type})

        assert [r["model_type"] for r in history.load_history()] == [
            "logreg",
            "random_forest",
            "logreg",
        ]

    def test_corrupted_file_is_treated_as_empty(self):
        history.HISTORY_PATH.write_text("not json", encoding="utf-8")

        assert history.load_history() == []

    def test_history_is_capped(self, monkeypatch):
        monkeypatch.setattr(history, "MAX_RECORDS", 3)

        for index in range(5):
            history.append_record({"run": index})

        assert [r["run"] for r in history.load_history()] == [2, 3, 4]


class TestModelMetricsEndpoint:
    def test_empty_history(self, client):
        assert client.get("/model/metrics").json() == {
            "latest": None,
            "history": [],
            "best": None,
            "total": 0,
        }

    def test_latest_history_and_best(self, client):
        client.post("/model/train")
        client.post("/model/train", json={"model_type": "random_forest"})

        data = client.get("/model/metrics").json()

        assert data["total"] == 2
        assert data["latest"]["model_type"] == "random_forest"
        assert len(data["history"]) == 2
        assert data["history"][0]["model_type"] == "logreg"
        assert data["best"] is not None

    def test_limit(self, client):
        for _ in range(3):
            client.post("/model/train")

        data = client.get("/model/metrics", params={"limit": 2}).json()

        assert data["total"] == 3
        assert len(data["history"]) == 2

    def test_filter_by_model_type(self, client):
        for model_type in ("logreg", "random_forest", "logreg"):
            client.post("/model/train", json={"model_type": model_type})

        data = client.get("/model/metrics", params={"model_type": "logreg"}).json()
        assert data["total"] == 2
        assert len(data["history"]) == 2
        assert all(r["model_type"] == "logreg" for r in data["history"])

        missing = client.get("/model/metrics", params={"model_type": "svm"}).json()
        assert missing["total"] == 0
        assert missing["latest"] is None

    def test_best_is_highest_f1_not_latest(self, client):
        records = [
            {
                "trained_at": "2026-01-01T00:00:00+00:00",
                "model_type": "logreg",
                "metrics": {"accuracy": 0.9, "f1": 0.9},
            },
            {
                "trained_at": "2026-01-02T00:00:00+00:00",
                "model_type": "random_forest",
                "metrics": {"accuracy": 0.6, "f1": 0.5},
            },
        ]
        history.HISTORY_PATH.write_text(json.dumps(records), encoding="utf-8")

        data = client.get("/model/metrics", params={"limit": 1}).json()

        assert data["latest"]["model_type"] == "random_forest"
        assert data["best"]["model_type"] == "logreg"
        assert len(data["history"]) == 1

    def test_invalid_limit(self, client):
        response = client.get("/model/metrics", params={"limit": 0})

        assert response.status_code == 422
        assert response.json()["code"] == "validation_error"
