# Churn Prediction API

FastAPI-сервис для предсказания оттока клиентов (churn prediction) на основе их поведения и характеристик.

## Задача

Клиент может уйти из сервиса в следующем месяце или остаться. Целевая переменная `churn`:
- **1** — клиент ушёл
- **0** — клиент остался

## Признаки в датасете

| Признак | Описание | Тип |
|---|---|---|
| `monthly_fee` | Ежемесячная стоимость тарифа | float |
| `usage_hours` | Часы использования сервиса за месяц | float |
| `support_requests` | Обращения в техподдержку | int |
| `account_age_months` | Возраст аккаунта в месяцах | int |
| `failed_payments` | Неудачные платежи | int |
| `region` | Регион клиента (europe, asia, america, africa) | str |
| `device_type` | Тип устройства (mobile, desktop, tablet) | str |
| `payment_method` | Способ оплаты (card, paypal, crypto) | str |
| `autopay_enabled` | Автосписание (0 или 1) | int |
| `churn` | Целевой признак (0/1) | int |

## Быстрый старт

### Установка

```bash
uv sync
```

### Запуск сервера

```bash
uv run churn-fastapi
```

Сервер поднимется на `http://localhost:8000`. Документация доступна по адресу `http://localhost:8000/docs`.

### Обучение модели

```bash
curl -X POST http://localhost:8000/train
```

### Получение предсказания

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_fee": 19.99,
    "usage_hours": 15.0,
    "support_requests": 2,
    "account_age_months": 6,
    "failed_payments": 0,
    "region": "europe",
    "device_type": "desktop",
    "payment_method": "card",
    "autopay_enabled": 1
  }'
```

## API эндпоинты

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/health` | Проверка здоровья сервиса |
| `POST` | `/train` | Обучение модели на загруженном датасете |
| `POST` | `/predict` | Предсказание для одного клиента |
| `POST` | `/predict/batch` | Пакетное предсказание |
| `GET` | `/metrics` | Метрики качества модели |
| `POST` | `/upload` | Загрузка CSV-датасета |
| `GET` | `/dataset` | Информация о текущем датасете |
| `GET` | `/dataset/info` | Размер датасета, признаки, распределение `churn` |
| `GET` | `/dataset/preview?limit=10` | Первые N строк датасета в JSON |
| `GET` | `/dataset/split-info` | Размеры train/test и распределение `churn` в выборках |

## Docker

```bash
docker compose up --build
```

## Тесты

```bash
uv run pytest -v
```

## Структура проекта

```
churn_fastapi/
├── data/
│   └── churn_dataset.csv      # Тренировочный датасет (2000 строк)
├── pyproject.toml             # Метаданные и зависимости
├── Dockerfile
├── docker-compose.yml
├── src/
│   └── churn_fastapi/
│       ├── __init__.py
│       ├── main.py            # FastAPI приложение и эндпоинты
│       ├── schemas.py         # Pydantic-модели запросов/ответов
│       ├── dataset.py         # Загрузка и предпросмотр датасета
│       ├── preprocessing.py   # X/y, пропуски, числовые/категориальные, train/test
│       ├── model.py           # Обучение, сохранение, загрузка модели
│       └── config.py          # Конфигурация и пути
├── models/                    # Сохранённые модели и метрики
└── tests/
    ├── conftest.py            # Фикстуры для тестов
    ├── test_api.py            # Тесты API
    ├── test_dataset.py        # Тесты загрузки и просмотра датасета
    └── test_preprocessing.py  # Тесты предобработки и разбиения
```

## Признаки по типам

Числовые и категориальные признаки заданы явно в `src/churn_fastapi/config.py`
(`NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`) — эндпоинт `GET /dataset/split-info`
покажет, что попало в каждую группу.

Пропуски обрабатываются внутри пайплайна (`SimpleImputer`): числовые — медианой,
категориальные — модой. Импутеры обучаются **только на train**, поэтому
тестовая выборка не участвует в подборе значений заполнения.

## Стек

- **FastAPI** — веб-фреймворк
- **scikit-learn** — обучение модели (RandomForestClassifier)
- **pandas** — работа с данными
- **joblib** — сохранение/загрузка модели
- **Pydantic** — валидация данных
- **uvicorn** — ASGI-сервер
