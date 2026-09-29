import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn_fastapi import dataset, model
from churn_fastapi.config import ALL_FEATURES, MODEL_PATH, TARGET_COLUMN

HEADER = ",".join(ALL_FEATURES + [TARGET_COLUMN]) + "\n"


@pytest.fixture
def clean_df():
    return dataset.load_dataframe().copy()


class TestTrainChurnModel:
    def test_returns_trained_pipeline(self, clean_df):
        result = model.train_churn_model(clean_df)
        assert isinstance(result, model.TrainedChurnModel)
        assert isinstance(result.pipeline, Pipeline)
        assert hasattr(result.pipeline, "predict")

    def test_pipeline_structure(self, clean_df):
        pipeline = model.train_churn_model(clean_df).pipeline
        steps = pipeline.named_steps
        assert "preprocessor" in steps
        assert "classifier" in steps
        assert isinstance(steps["classifier"], LogisticRegression)

        num_steps = steps["preprocessor"].named_transformers_["num"].named_steps
        cat_steps = steps["preprocessor"].named_transformers_["cat"].named_steps
        assert "scaler" in num_steps
        assert "imputer" in num_steps
        assert "onehot" in cat_steps

    def test_numeric_scaled_categorical_encoded(self, clean_df):
        pipeline = model.train_churn_model(clean_df).pipeline
        pre = pipeline.named_steps["preprocessor"]
        assert list(pre.named_transformers_["num"].named_steps["scaler"].get_feature_names_out())
        onehot = pre.named_transformers_["cat"].named_steps["onehot"]
        assert onehot.handle_unknown == "ignore"

    def test_split_sizes_and_metrics(self, clean_df):
        result = model.train_churn_model(clean_df)
        assert result.split.train_size == 1600
        assert result.split.test_size == 400
        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert 0.0 <= result.metrics["f1"] <= 1.0
        assert result.metrics["train_size"] == 1600
        assert result.metrics["test_size"] == 400

    def test_model_name(self):
        assert model.MODEL_NAME == "LogisticRegression"

    def test_custom_split_params(self, clean_df):
        result = model.train_churn_model(clean_df, test_size=0.3, random_state=7)
        assert result.split.test_size == 600
        assert result.split.train_size == 1400

    def test_is_deterministic(self, clean_df):
        a = model.train_churn_model(clean_df)
        b = model.train_churn_model(clean_df)
        assert a.metrics == b.metrics

    def test_empty_dataframe_raises(self):
        with pytest.raises(model.InvalidDatasetError, match="пуст"):
            model.train_churn_model(pd.DataFrame(columns=ALL_FEATURES + [TARGET_COLUMN]))

    def test_missing_columns_raise(self, clean_df):
        with pytest.raises(model.InvalidDatasetError, match="отсутствуют"):
            model.train_churn_model(clean_df.drop(columns=["region", TARGET_COLUMN]))

    def test_single_class_raises(self, clean_df):
        only_one = clean_df[clean_df[TARGET_COLUMN] == 1]
        with pytest.raises(model.InvalidDatasetError, match="оба класса"):
            model.train_churn_model(only_one)

    def test_invalid_test_size_raises(self, clean_df):
        with pytest.raises(ValueError, match="test_size"):
            model.train_churn_model(clean_df, test_size=1.5)


class TestModelTrainEndpoint:
    def test_returns_accuracy_and_f1(self, client):
        resp = client.post("/model/train")
        assert resp.status_code == 200
        data = resp.json()
        assert data["message"] == "Модель обучена успешно"
        assert data["model"] == "LogisticRegression"
        assert 0.0 <= data["accuracy"] <= 1.0
        assert 0.0 <= data["f1"] <= 1.0
        assert data["train_size"] == 1600
        assert data["test_size"] == 400

    def test_legacy_train_endpoint_is_alias(self, client):
        legacy = client.post("/train")
        canonical = client.post("/model/train")
        assert legacy.status_code == canonical.status_code == 200
        assert legacy.json() == canonical.json()

    def test_model_is_usable_after_training(self, client, sample_client):
        client.post("/model/train")
        resp = client.post("/predict", json=sample_client)
        assert resp.status_code == 200
        data = resp.json()
        assert data["churn"] in (0, 1)
        assert 0.0 <= data["probability"] <= 1.0

    def test_metrics_saved_to_file(self, client):
        client.post("/model/train")
        metrics = model.load_metrics()
        assert metrics is not None
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert MODEL_PATH.exists()

    def test_404_when_dataset_missing(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(dataset, "DATASET_PATH", tmp_path / "missing.csv")
        dataset.clear_cache()
        try:
            resp = client.post("/model/train")
            assert resp.status_code == 404
            assert "не найден" in resp.json()["detail"]
        finally:
            dataset.clear_cache()

    def test_400_when_dataset_empty(self, client, monkeypatch, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text(HEADER, encoding="utf-8")
        monkeypatch.setattr(dataset, "DATASET_PATH", empty)
        dataset.clear_cache()
        try:
            resp = client.post("/model/train")
            assert resp.status_code == 400
            assert "пуст" in resp.json()["detail"]
        finally:
            dataset.clear_cache()

    def test_400_when_dataset_has_one_class(self, client, monkeypatch, tmp_path):
        row = (
            "9.99,27.92,1,14,1,america,desktop,card,1,1\n"
            "19.99,21.48,2,1,0,america,mobile,card,1,1\n"
        )
        one_class = tmp_path / "one_class.csv"
        one_class.write_text(HEADER + row, encoding="utf-8")
        monkeypatch.setattr(dataset, "DATASET_PATH", one_class)
        dataset.clear_cache()
        try:
            resp = client.post("/model/train")
            assert resp.status_code == 400
            assert "оба класса" in resp.json()["detail"]
        finally:
            dataset.clear_cache()

    def test_400_when_columns_missing(self, client, monkeypatch, tmp_path):
        broken = tmp_path / "broken.csv"
        broken.write_text("a,b,c\n1,2,3\n1,2,3\n", encoding="utf-8")
        monkeypatch.setattr(dataset, "DATASET_PATH", broken)
        dataset.clear_cache()
        try:
            resp = client.post("/model/train")
            assert resp.status_code == 400
            assert "отсутствуют" in resp.json()["detail"]
        finally:
            dataset.clear_cache()
