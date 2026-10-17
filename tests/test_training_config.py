import json

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from churn_fastapi import dataset, model
from churn_fastapi.config import MODEL_METADATA_PATH
from churn_fastapi.schemas import TrainingConfigChurn


@pytest.fixture(autouse=True)
def clean_model_state():
    model.reset_model()
    yield
    model.reset_model()


@pytest.fixture
def clean_df():
    return dataset.load_dataframe().copy()


class TestTrainingConfigChurn:
    def test_defaults(self):
        config = TrainingConfigChurn()
        assert config.model_type == "logreg"
        assert config.hyperparameters == {}

    def test_custom_values(self):
        config = TrainingConfigChurn(
            model_type="random_forest", hyperparameters={"n_estimators": 100}
        )
        assert config.model_type == "random_forest"
        assert config.hyperparameters == {"n_estimators": 100}


class TestBuildClassifier:
    def test_default_is_logreg(self):
        clf = model.build_classifier()
        assert isinstance(clf, LogisticRegression)
        assert clf.max_iter == 1000
        assert clf.random_state == 42

    def test_logreg_with_hyperparameters(self):
        clf = model.build_classifier(TrainingConfigChurn(hyperparameters={"C": 0.5}))
        assert isinstance(clf, LogisticRegression)
        assert clf.C == 0.5

    def test_random_forest(self):
        clf = model.build_classifier(
            TrainingConfigChurn(model_type="random_forest", hyperparameters={"n_estimators": 25})
        )
        assert isinstance(clf, RandomForestClassifier)
        assert clf.n_estimators == 25
        assert clf.random_state == 42

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Неизвестный тип модели"):
            model.build_classifier(TrainingConfigChurn(model_type="svm"))

    def test_invalid_hyperparameter_raises(self):
        with pytest.raises(TypeError):
            model.build_classifier(TrainingConfigChurn(hyperparameters={"nonexistent": 1}))


class TestTrainWithConfig:
    def test_random_forest_pipeline(self, clean_df):
        config = TrainingConfigChurn(
            model_type="random_forest", hyperparameters={"n_estimators": 20}
        )
        result = model.train_churn_model(clean_df, config=config)
        assert isinstance(result.pipeline.named_steps["classifier"], RandomForestClassifier)
        assert result.model_type == "random_forest"
        assert result.hyperparameters["n_estimators"] == 20
        assert 0.0 <= result.metrics["accuracy"] <= 1.0

    def test_default_is_logreg(self, clean_df):
        result = model.train_churn_model(clean_df)
        assert isinstance(result.pipeline.named_steps["classifier"], LogisticRegression)
        assert result.model_type == "logreg"

    def test_metadata_contains_model_type_and_params(self, clean_df):
        config = TrainingConfigChurn(model_type="random_forest")
        result = model.train_churn_model(clean_df, config=config)
        from churn_fastapi.model import _build_metadata

        metadata = _build_metadata(result, clean_df)
        assert metadata["model"] == "RandomForestClassifier"
        assert metadata["model_type"] == "random_forest"
        assert metadata["hyperparameters"]["random_state"] == 42
        json.dumps(metadata, default=str)


class TestTrainEndpointWithConfig:
    def test_default_body_still_works(self, client):
        resp = client.post("/model/train")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "LogisticRegression"
        assert data["model_type"] == "logreg"
        assert data["hyperparameters"]["max_iter"] == 1000

    def test_train_random_forest(self, client):
        resp = client.post(
            "/model/train",
            json={"model_type": "random_forest", "hyperparameters": {"n_estimators": 30}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "RandomForestClassifier"
        assert data["model_type"] == "random_forest"
        assert data["hyperparameters"]["n_estimators"] == 30
        assert 0.0 <= data["accuracy"] <= 1.0

    def test_legacy_endpoint_accepts_config(self, client):
        payload = {"model_type": "random_forest", "hyperparameters": {"n_estimators": 10}}
        legacy = client.post("/train", json=payload)
        canonical = client.post("/model/train", json=payload)
        assert legacy.status_code == canonical.status_code == 200
        assert legacy.json() == canonical.json()

    def test_unknown_model_type_returns_400(self, client):
        resp = client.post("/model/train", json={"model_type": "svm"})
        assert resp.status_code == 400
        assert "Неизвестный тип модели" in resp.json()["detail"]

    def test_invalid_hyperparameter_returns_400(self, client):
        resp = client.post("/model/train", json={"hyperparameters": {"nonexistent": 1}})
        assert resp.status_code == 400
        assert "Не удалось обучить модель" in resp.json()["detail"]


class TestStatusShowsModelType:
    def test_status_after_random_forest_training(self, client):
        client.post("/model/train", json={"model_type": "random_forest"})
        data = client.get("/model/status").json()
        assert data["model"] == "RandomForestClassifier"
        assert data["model_type"] == "random_forest"
        assert data["hyperparameters"]["n_estimators"] == 100

    def test_status_before_training_has_no_model_type(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(model, "MODEL_PATH", tmp_path / "absent.joblib")
        monkeypatch.setattr(model, "MODEL_METADATA_PATH", tmp_path / "absent.json")
        model.reset_model()
        data = client.get("/model/status").json()
        assert data["model_type"] is None
        assert data["hyperparameters"] is None

    def test_model_type_saved_to_metadata_file(self, client):
        client.post(
            "/model/train",
            json={"model_type": "random_forest", "hyperparameters": {"max_depth": 7}},
        )
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        assert metadata["model_type"] == "random_forest"
        assert metadata["hyperparameters"]["max_depth"] == 7
