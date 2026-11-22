import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from churn_fastapi.api.errors import register_error_handlers
from churn_fastapi.api.routes import router
from churn_fastapi.ml.model import load_model, reset_model

logger = logging.getLogger("uvicorn.error.churn_fastapi.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if load_model() is not None:
        logger.info("Приложение стартовало с моделью, загруженной с диска")
    else:
        logger.info("Сохранённая модель не найдена — обучите через POST /model/train")
    yield
    reset_model()


app = FastAPI(
    title="Churn Prediction API",
    description="Сервис для предсказания оттока клиентов",
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Prediction",
            "description": (
                "Предсказание оттока (churn) клиентов. "
                "Эндпоинт /predict принимает один объект FeatureVectorChurn "
                "или список таких объектов."
            ),
        },
    ],
)
app.include_router(router)
register_error_handlers(app)


def run():
    import uvicorn

    uvicorn.run("churn_fastapi.main:app", host="0.0.0.0", port=8000, reload=True)
