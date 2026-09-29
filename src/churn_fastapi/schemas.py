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


class SplitInfoResponse(BaseModel):
    test_size: float = Field(..., description="Доля тестовой выборки")
    random_state: int = Field(..., description="Seed для воспроизводимости")
    stratify: bool = Field(..., description="Стратификация по churn")
    total_rows: int = Field(..., description="Всего строк в датасете")
    train_size: int = Field(..., description="Строк в train")
    test_rows: int = Field(..., description="Строк в test")
    numeric_features: list[str] = Field(..., description="Числовые признаки")
    categorical_features: list[str] = Field(..., description="Категориальные признаки")
    features: list[str] = Field(..., description="Все признаки матрицы X")
    target_column: str = Field(..., description="Целевая переменная")
    missing_values: dict[str, int] = Field(..., description="Пропуски по столбцам")
    missing_values_total: int = Field(..., description="Всего пропусков")
    churn_distribution: dict[str, dict[str, int]] = Field(
        ..., description="Распределение churn для full / train / test"
    )
    churn_ratio: dict[str, float] = Field(..., description="Доля churn = 1 в выборках")
    stratified_consistent: bool = Field(
        ..., description="Распределение churn в train и test примерно одинаковое"
    )
