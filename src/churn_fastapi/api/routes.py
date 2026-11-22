import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Query, UploadFile

from churn_fastapi.core.config import ALL_FEATURES, DATASET_PATH, TARGET_COLUMN
from churn_fastapi.core.errors import APIError
from churn_fastapi.ml import dataset, preprocessing
from churn_fastapi.ml.history import get_training_metrics
from churn_fastapi.ml.model import (
    InvalidDatasetError,
    get_metadata,
    get_model_paths,
    is_loaded_from_disk,
    load_metrics,
    load_model,
    predict,
    train,
)
from churn_fastapi.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DatasetInfoResponse,
    DatasetPreviewResponse,
    FeatureSpecChurn,
    FeatureVectorChurn,
    HealthResponse,
    MetricsResponse,
    ModelSchemaResponse,
    ModelStatusResponse,
    ModelTrainResponse,
    PredictionResponseChurn,
    SplitInfoResponse,
    TrainingConfigChurn,
    TrainingMetricsResponse,
)

logger = logging.getLogger("uvicorn.error.churn_fastapi.api")
router = APIRouter()


@router.get("/")
def root():
    return {"message": "ml churn service is running"}


@router.get("/health", response_model=HealthResponse)
def health():
    try:
        dataset.load_dataframe()
        dataset_loaded = True
    except (OSError, ValueError) as e:
        logger.warning("Датасет недоступен для /health: %s", e)
        dataset_loaded = False

    return HealthResponse(
        status="ok",
        model_loaded=load_model() is not None,
        dataset_loaded=dataset_loaded,
    )


@router.post("/model/train", response_model=ModelTrainResponse)
def train_churn_model_endpoint(config: TrainingConfigChurn | None = None):
    try:
        metadata = train(config=config)
    except FileNotFoundError as e:
        raise APIError(
            status_code=404,
            detail=str(e) or "Файл датасета не найден",
            code="dataset_not_found",
        )
    except InvalidDatasetError as e:
        code = "empty_dataset" if "пуст" in str(e).lower() else "invalid_dataset"
        raise APIError(status_code=400, detail=str(e), code=code)
    except (ValueError, TypeError) as e:
        raise APIError(
            status_code=400,
            detail=f"Не удалось обучить модель: {e}",
            code="invalid_model_config",
        )
    metrics = metadata["metrics"]
    return ModelTrainResponse(
        message="Модель обучена успешно",
        model=metadata["model"],
        model_type=metadata["model_type"],
        hyperparameters=metadata["hyperparameters"],
        accuracy=metrics["accuracy"],
        f1=metrics["f1"],
        train_size=metrics["train_size"],
        test_size=metrics["test_size"],
    )


@router.post("/train", response_model=ModelTrainResponse)
def train_model(config: TrainingConfigChurn | None = None):
    return train_churn_model_endpoint(config)


@router.get("/model/status", response_model=ModelStatusResponse)
def model_status():
    pipeline = load_model()
    metadata = get_metadata()
    model_path, metadata_path = get_model_paths()
    trained_at = None
    age_seconds = None
    if metadata and metadata.get("trained_at"):
        try:
            trained_dt = datetime.fromisoformat(metadata["trained_at"])
            trained_at = trained_dt.isoformat()
            age_seconds = round(
                (datetime.now(timezone.utc) - trained_dt).total_seconds(), 2
            )
        except ValueError:
            logger.warning("Не удалось разобрать trained_at: %s", metadata.get("trained_at"))

    raw_metrics = metadata.get("metrics") if metadata else None
    metrics = MetricsResponse(**raw_metrics) if raw_metrics else None

    return ModelStatusResponse(
        trained=pipeline is not None,
        model=metadata.get("model") if metadata else None,
        model_type=metadata.get("model_type") if metadata else None,
        hyperparameters=metadata.get("hyperparameters") if metadata else None,
        trained_at=trained_at,
        age_seconds=age_seconds,
        metrics=metrics,
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        model_file_exists=model_path.exists(),
        loaded_from_disk=is_loaded_from_disk(),
        features=list(ALL_FEATURES),
        dataset_rows=metadata.get("dataset_rows") if metadata else None,
        sklearn_version=metadata.get("sklearn_version") if metadata else None,
    )


@router.get("/model/schema", response_model=ModelSchemaResponse)
def model_schema():
    json_schema = FeatureVectorChurn.model_json_schema()
    required = set(json_schema.get("required", []))
    numeric, categorical = preprocessing.get_column_groups()
    features = []
    for name in ALL_FEATURES:
        prop = json_schema["properties"][name]
        features.append(
            FeatureSpecChurn(
                name=name,
                type=prop["type"],
                group="numeric" if name in numeric else "categorical",
                required=name in required,
                description=prop.get("description"),
                minimum=prop.get("minimum"),
                maximum=prop.get("maximum"),
            )
        )
    return ModelSchemaResponse(
        features=features,
        numeric_features=numeric,
        categorical_features=categorical,
        target_column=TARGET_COLUMN,
    )


@router.post(
    "/predict",
    response_model=PredictionResponseChurn | list[PredictionResponseChurn],
    responses={
        400: {"description": "Модель не обучена или некорректный запрос"},
    },
    tags=["Prediction"],
)
def predict_endpoint(payload: FeatureVectorChurn | list[FeatureVectorChurn]):
    count = len(payload) if isinstance(payload, list) else 1
    logger.info("POST /predict: %d записей", count)

    if isinstance(payload, list):
        df = preprocessing.build_feature_matrix([c.model_dump() for c in payload])
        try:
            preds, probas = predict(df)
        except RuntimeError as e:
            raise APIError(status_code=400, detail=str(e), code="model_not_trained")
        logger.info("POST /predict: получено %d предсказаний", len(preds))
        return [
            PredictionResponseChurn(churn=p, probability=round(pr, 4))
            for p, pr in zip(preds, probas)
        ]
    df = preprocessing.build_feature_matrix(payload.model_dump())
    try:
        preds, probas = predict(df)
    except RuntimeError as e:
        raise APIError(status_code=400, detail=str(e), code="model_not_trained")
    logger.info("POST /predict: churn=%s, probability=%.4f", preds[0], probas[0])
    return PredictionResponseChurn(churn=preds[0], probability=round(probas[0], 4))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    df = preprocessing.build_feature_matrix([c.model_dump() for c in request.clients])
    try:
        preds, probas = predict(df)
    except RuntimeError as e:
        raise APIError(status_code=400, detail=str(e), code="model_not_trained")
    results = [
        PredictionResponseChurn(churn=p, probability=round(pr, 4))
        for p, pr in zip(preds, probas)
    ]
    return BatchPredictionResponse(predictions=results)


@router.get("/metrics", response_model=MetricsResponse)
def metrics():
    m = load_metrics()
    if m is None:
        raise APIError(
            status_code=404,
            detail="Метрики отсутствуют. Сначала обучите модель",
            code="metrics_not_available",
        )
    return MetricsResponse(**m)


@router.get("/model/metrics", response_model=TrainingMetricsResponse)
def model_metrics(
    limit: int = Query(10, ge=1, le=100, description="Сколько последних обучений вернуть"),
    model_type: str | None = Query(None, description="Фильтр по типу модели"),
):
    return TrainingMetricsResponse(
        **get_training_metrics(limit=limit, model_type=model_type)
    )


@router.post("/upload")
async def upload_dataset(file: UploadFile):
    if not file.filename or not file.filename.endswith(".csv"):
        raise APIError(status_code=400, detail="Требуется CSV-файл", code="invalid_file")
    content = await file.read()
    DATASET_PATH.write_bytes(content)
    dataset.clear_cache()
    return {"message": f"Датасет {file.filename} загружен", "rows": content.count(b"\n") - 1}


def _load_dataset_df():
    try:
        return dataset.load_dataframe()
    except dataset.DatasetNotFoundError:
        raise APIError(status_code=404, detail="Датасет не найден", code="dataset_not_found")
    except ValueError as e:
        raise APIError(status_code=422, detail=str(e), code="validation_error")


@router.get("/dataset/info", response_model=DatasetInfoResponse)
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


@router.get("/dataset", response_model=dict)
def dataset_info_legacy():
    df = _load_dataset_df()
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "churn_distribution": dataset.get_churn_distribution(),
    }


@router.get("/dataset/preview", response_model=DatasetPreviewResponse)
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


@router.get("/dataset/split-info", response_model=SplitInfoResponse)
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
        raise APIError(status_code=422, detail=str(e), code="validation_error")
    return SplitInfoResponse(
        **preprocessing.summarize_split(
            split,
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    )
