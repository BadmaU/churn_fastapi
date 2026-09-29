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


class DatasetRowChurn(BaseModel):
    monthly_fee: float = Field(..., description="Ежемесячная стоимость тарифа")
    usage_hours: float = Field(..., description="Часы использования за месяц")
    support_requests: int = Field(..., description="Обращения в техподдержку")
    account_age_months: int = Field(..., description="Возраст аккаунта в месяцах")
    failed_payments: int = Field(..., description="Неудачные платежи")
    region: str = Field(..., description="Регион клиента")
    device_type: str = Field(..., description="Тип устройства")
    payment_method: str = Field(..., description="Способ оплаты")
    autopay_enabled: int = Field(..., description="Автосписание: 0 или 1")
    churn: int = Field(..., description="Целевой признак: 1 — ушёл, 0 — остался")


class DatasetPreviewResponse(BaseModel):
    total_rows: int = Field(..., description="Всего строк в датасете")
    returned: int = Field(..., description="Сколько строк вернулось")
    limit: int = Field(..., description="Запрошенный лимит строк")
    rows: list[DatasetRowChurn]


class DatasetInfoResponse(BaseModel):
    path: str = Field(..., description="Путь к файлу датасета")
    exists: bool = Field(..., description="Существует ли файл")
    rows: int = Field(..., description="Количество строк")
    columns: int = Field(..., description="Количество столбцов")
    features: list[str] = Field(..., description="Список названий признаков")
    target_column: str = Field(..., description="Название целевой переменной")
    churn_distribution: dict[str, int] = Field(..., description="Распределение churn по классам")
    churn_ratio: float = Field(..., description="Доля ушедших клиентов (churn = 1)")
