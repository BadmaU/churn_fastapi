def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_train_and_predict(client, sample_client):
    resp = client.post("/train")
    assert resp.status_code == 200
    assert resp.json()["message"] == "Модель обучена успешно"

    resp = client.post("/predict", json=sample_client)
    assert resp.status_code == 200
    data = resp.json()
    assert data["churn"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0


def test_metrics(client):
    client.post("/train")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "accuracy" in data
    assert "f1" in data
    assert "roc_auc" in data


def test_batch_predict(client, sample_client):
    client.post("/train")
    batch = {"clients": [sample_client, sample_client]}
    resp = client.post("/predict/batch", json=batch)
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 2


def test_predict_without_model(client, sample_client, monkeypatch, tmp_path):
    from churn_fastapi.ml import model as m

    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "absent.joblib")
    m.reset_model()

    resp = client.post("/predict", json=sample_client)
    assert resp.status_code == 400
    assert resp.json()["code"] == "model_not_trained"


def test_dataset_info(client):
    resp = client.get("/dataset")
    assert resp.status_code == 200
    data = resp.json()
    assert data["rows"] == 2000
    assert "churn" in data["columns"]


def test_invalid_predict(client):
    resp = client.post("/predict", json={"monthly_fee": -1})
    assert resp.status_code == 422
