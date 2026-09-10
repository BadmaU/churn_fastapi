from pydantic import BaseModel, Field


class ClientFeatures(BaseModel):
    monthly_fee: float = Field(..., ge=0, description="Ежемесячная стоимость тарифа")
    usage_hours: float = Field(..., ge=0, description="Часы использования за месяц")
    support_requests: int = Field(..., ge=0, description="Обращения в техподдержку")
    account_age_months: int = Field(..., ge=1, description="Возраст аккаунта в месяцах")
    failed_payments: int = Field(..., ge=0, description="Неудачные платежи")
    region: str = Field(..., description="Регион: europe, asia, america, africa")
    device_type: str = Field(..., description="Устройство: mobile, desktop, tablet")
    payment_method: str = Field(..., description="Оплата: card, paypal, crypto")
    autopay_enabled: int = Field(..., ge=0, le=1, description="Автосписание: 0 или 1")


class PredictionResponse(BaseModel):
    churn: int = Field(..., description="Предсказание: 1 — ушёл, 0 — остался")
    probability: float = Field(..., description="Вероятность оттока")


class BatchPredictionRequest(BaseModel):
    clients: list[ClientFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    train_size: int
    test_size: int


class TrainResponse(BaseModel):
    message: str
    metrics: MetricsResponse


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
