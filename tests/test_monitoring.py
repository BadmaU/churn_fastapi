import logging

from churn_fastapi.ml import dataset


def test_health_reports_dataset_and_model(client, synthetic_csv, isolated_model):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["dataset_loaded"] is True
    assert body["model_loaded"] is False


def test_health_reports_missing_dataset(client, monkeypatch, tmp_path):
    monkeypatch.setattr(dataset, "DATASET_PATH", tmp_path / "absent.csv")
    dataset.clear_cache()
    try:
        response = client.get("/health")
        assert response.json()["dataset_loaded"] is False
    finally:
        dataset.clear_cache()


def test_dataset_load_is_logged(caplog, synthetic_csv):
    caplog.set_level(logging.INFO)
    dataset.clear_cache()
    dataset.load_dataframe()

    assert "Датасет загружен" in caplog.text


def test_training_and_prediction_are_logged(
    client, sample_client, caplog, synthetic_csv, isolated_model
):
    caplog.set_level(logging.INFO)

    client.post("/model/train")
    client.post("/predict", json=sample_client)

    assert "Обучение модели" in caplog.text
    assert "Обучение завершено" in caplog.text
    assert "POST /predict" in caplog.text


def test_http_error_is_logged(client, caplog):
    caplog.set_level(logging.WARNING)

    response = client.get("/no-such-route")

    assert response.status_code == 404
    assert "Ошибка not_found" in caplog.text


def test_validation_error_is_logged(client, caplog):
    caplog.set_level(logging.WARNING)

    response = client.post("/predict", json={"monthly_fee": "not-a-number"})

    assert response.status_code == 422
    assert "Ошибка validation_error" in caplog.text
