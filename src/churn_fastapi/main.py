from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, UploadFile

from churn_fastapi import dataset, preprocessing
from churn_fastapi.config import ALL_FEATURES, DATASET_PATH, TARGET_COLUMN
from churn_fastapi.model import (
    MODEL_NAME,
    InvalidDatasetError,
    load_metrics,
    load_model,
    predict,
    train,
)
from churn_fastapi.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ClientFeatures,
    DatasetInfoResponse,
    DatasetPreviewResponse,
    HealthResponse,
    MetricsResponse,
    ModelTrainResponse,
    PredictionResponse,
    SplitInfoResponse,
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


@app.post("/model/train", response_model=ModelTrainResponse)
def train_churn_model_endpoint():
    try:
        metrics = train()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файл датасета не найден")
    except InvalidDatasetError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Не удалось обучить модель: {e}")
    return ModelTrainResponse(
        message="Модель обучена успешно",
        model=MODEL_NAME,
        accuracy=metrics["accuracy"],
        f1=metrics["f1"],
        train_size=metrics["train_size"],
        test_size=metrics["test_size"],
    )


@app.post("/train", response_model=ModelTrainResponse)
def train_model():
    return train_churn_model_endpoint()


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
    dataset.clear_cache()
    return {"message": f"Датасет {file.filename} загружен", "rows": content.count(b"\n") - 1}


def _load_dataset_df():
    try:
        return dataset.load_dataframe()
    except dataset.DatasetNotFoundError:
        raise HTTPException(status_code=404, detail="Датасет не найден")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/dataset/info", response_model=DatasetInfoResponse)
def dataset_info():
    df = _load_dataset_df()
    return DatasetInfoResponse(
        path=str(DATASET_PATH),
        exists=True,
        rows=len(df),
        columns=len(df.columns),
        features=dataset.get_features(),
        target_column=TARGET_COLUMN,
        churn_distribution=dataset.get_churn_distribution(),
        churn_ratio=dataset.get_churn_ratio(),
    )


@app.get("/dataset", response_model=dict)
def dataset_info_legacy():
    df = _load_dataset_df()
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "churn_distribution": dataset.get_churn_distribution(),
    }


@app.get("/dataset/preview", response_model=DatasetPreviewResponse)
def dataset_preview(
    limit: int = Query(10, ge=1, le=10000, description="Сколько строк вернуть"),
):
    _load_dataset_df()
    rows = dataset.get_rows(limit)
    return DatasetPreviewResponse(
        total_rows=dataset.count_rows(),
        returned=len(rows),
        limit=limit,
        rows=rows,
    )


@app.get("/dataset/split-info", response_model=SplitInfoResponse)
def dataset_split_info(
    test_size: float = Query(
        preprocessing.DEFAULT_TEST_SIZE, gt=0, lt=1, description="Доля тестовой выборки"
    ),
    random_state: int = Query(preprocessing.DEFAULT_RANDOM_STATE, description="Seed"),
    stratify: bool = Query(True, description="Стратифицировать по churn"),
):
    df = _load_dataset_df()
    try:
        split = preprocessing.make_split(
            df, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return SplitInfoResponse(
        **preprocessing.summarize_split(
            split,
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    )


def run():
    import uvicorn
    uvicorn.run("churn_fastapi.main:app", host="0.0.0.0", port=8000, reload=True)
