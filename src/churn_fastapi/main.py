from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile

from churn_fastapi.config import ALL_FEATURES, DATASET_PATH
from churn_fastapi.model import load_metrics, load_model, predict, train
from churn_fastapi.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ClientFeatures,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
    TrainResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Сервис для предсказания оттока клиентов",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=load_model() is not None,
    )


@app.post("/train", response_model=TrainResponse)
def train_model():
    try:
        metrics = train()
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Файл датасета не найден")
    return TrainResponse(message="Модель обучена успешно", metrics=MetricsResponse(**metrics))


@app.post("/predict", response_model=PredictionResponse)
def predict_single(client: ClientFeatures):
    df = pd.DataFrame([client.model_dump()])
    df = df[ALL_FEATURES]
    try:
        preds, probas = predict(df)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PredictionResponse(churn=preds[0], probability=round(probas[0], 4))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    df = pd.DataFrame([c.model_dump() for c in request.clients])
    df = df[ALL_FEATURES]
    try:
        preds, probas = predict(df)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    results = [
        PredictionResponse(churn=p, probability=round(pr, 4))
        for p, pr in zip(preds, probas)
    ]
    return BatchPredictionResponse(predictions=results)


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    m = load_metrics()
    if m is None:
        raise HTTPException(status_code=404, detail="Метрики отсутствуют. Сначала обучите модель")
    return MetricsResponse(**m)


@app.post("/upload")
async def upload_dataset(file: UploadFile):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Требуется CSV-файл")
    content = await file.read()
    DATASET_PATH.write_bytes(content)
    return {"message": f"Датасет {file.filename} загружен", "rows": content.count(b"\n") - 1}


@app.get("/dataset", response_model=dict)
def dataset_info():
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Датасет не найден")
    df = pd.read_csv(DATASET_PATH)
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "churn_distribution": df["churn"].value_counts().to_dict(),
    }


def run():
    import uvicorn
    uvicorn.run("churn_fastapi.main:app", host="0.0.0.0", port=8000, reload=True)
