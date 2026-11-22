from churn_fastapi import dataset, model, preprocessing
from churn_fastapi.config import ALL_FEATURES, TARGET_COLUMN


class TestSyntheticPipelineWithoutFastAPI:
    def test_csv_has_expected_columns(self, synthetic_csv, synthetic_df):
        df = dataset.load_dataframe()

        assert list(df.columns) == ALL_FEATURES + [TARGET_COLUMN]
        assert len(df) == len(synthetic_df)
        assert set(df[TARGET_COLUMN].unique()) == {0, 1}

    def test_preprocessing_and_training(self, synthetic_df, sample_client):
        matrix = preprocessing.build_feature_matrix(sample_client)
        assert list(matrix.columns) == ALL_FEATURES

        split = preprocessing.make_split(synthetic_df)
        assert split.train_size + split.test_size == len(synthetic_df)

        result = model.train_churn_model(synthetic_df)
        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert 0.0 <= result.metrics["f1"] <= 1.0


class TestFullPipelineWithTestClient:
    def test_csv_train_status_predict(
        self, client, synthetic_csv, synthetic_df, isolated_model, sample_client
    ):
        df = dataset.load_dataframe()
        assert len(df) == len(synthetic_df)

        trained = client.post("/model/train")
        assert trained.status_code == 200
        train_data = trained.json()
        assert train_data["message"] == "Модель обучена успешно"

        status_response = client.get("/model/status")
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["trained"] is True
        assert status["model_type"] == train_data["model_type"]
        assert status["metrics"]["accuracy"] == train_data["accuracy"]

        prediction = client.post("/predict", json=sample_client)
        assert prediction.status_code == 200
        result = prediction.json()
        assert result["churn"] in (0, 1)
        assert 0.0 <= result["probability"] <= 1.0

    def test_training_is_reproducible(self, client, synthetic_csv, isolated_model):
        first = client.post("/model/train").json()
        second = client.post("/model/train").json()

        assert first["accuracy"] == second["accuracy"]
        assert first["f1"] == second["f1"]
        assert first["train_size"] == second["train_size"]

    def test_predict_without_trained_model(self, client, isolated_model, sample_client):
        response = client.post("/predict", json=sample_client)

        assert response.status_code == 400
        body = response.json()
        assert body["code"] == "model_not_trained"
        assert {"code", "message", "details"} <= body.keys()
