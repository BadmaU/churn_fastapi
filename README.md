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
curl -X POST http://localhost:8000/model/train
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
| `POST` | `/model/train` | Обучить LogisticRegression, вернуть accuracy и f1 |
| `POST` | `/train` | Алиас `/model/train` |
| `GET` | `/model/status` | Обучена ли модель, когда обучена, метрики |
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
│       ├── model.py           # train_churn_model, сохранение и загрузка модели
│       └── config.py          # Конфигурация и пути
├── models/                    # churn_model.joblib + model_metadata.json (в gitignore)
└── tests/
    ├── conftest.py            # Фикстуры для тестов
    ├── test_api.py            # Тесты API
    ├── test_dataset.py        # Тесты загрузки и просмотра датасета
    ├── test_preprocessing.py  # Тесты предобработки и разбиения
    └── test_model.py          # Тесты обучения модели
```

## Хранение модели

Обученная модель не теряется при перезапуске сервиса:

| Файл | Содержимое |
|---|---|
| `models/churn_model.joblib` | сам `Pipeline` через `joblib.dump` |
| `models/model_metadata.json` | `trained_at`, метрики, список признаков, версия sklearn, размер датасета |

- `save_churn_model(pipeline, metadata)` — пишет оба файла и обновляет модель в памяти
- `load_churn_model()` — читает модель с диска в глобальную переменную модуля `model.py`
- `reset_model()` — сбрасывает состояние (вызывается в `lifespan` при остановке)

`lifespan` при старте вызывает `load_churn_model()`, поэтому после перезапуска
сервис сразу предсказывает без повторного обучения. Повреждённый файл модели
не роняет приложение — в лог пишется warning, сервис стартует без модели.

Проверить: `GET /model/status` → поле `loaded_from_disk: true` означает, что
модель взята с диска, а не обучена в текущем процессе.

## Модель

`LogisticRegression(max_iter=1000, random_state=42)` в `Pipeline`:
`ColumnTransformer` (SimpleImputer + StandardScaler для числовых,
SimpleImputer + OneHotEncoder для категориальных) → классификатор.
Функция `train_churn_model(df)` принимает DataFrame и возвращает
`TrainedChurnModel` с полями `pipeline`, `split` и `metrics`.

### Качество базовой модели

Метрики на тестовой выборке (400 строк):

| Метрика | Значение |
|---|---|
| accuracy | 0.7875 |
| f1 | 0.0449 |
| recall | 0.0247 |
| roc_auc | 0.6091 |

**Это слабая модель, и причина в первую очередь в данных, а не в настройке.** Все признаки
слабо коррелируют с `churn` (максимум |r| = 0.136 для `usage_hours`
и `autopay_enabled`), поэтому roc_auc не поднимается выше ~0.61 ни одной
из проверенных моделей. Низкий f1 объясняется перекосом классов
(80% / 20%) и порогом по умолчанию 0.5: модель почти всегда предсказывает
«остался».

Проверено экспериментально:

| Конфигурация | accuracy | f1 | recall | roc_auc |
|---|---|---|---|---|
| LogisticRegression (текущая) | 0.7875 | 0.0449 | 0.0247 | 0.6091 |
| LogisticRegression `class_weight="balanced"` | 0.5900 | 0.3543 | 0.5556 | 0.6112 |
| RandomForest 100 деревьев | 0.7825 | 0.1714 | 0.1111 | 0.5847 |
| RandomForest `class_weight="balanced"` | 0.6950 | 0.2375 | 0.2346 | 0.5853 |

`class_weight="balanced"` поднимает f1 в 8 раз, но roc_auc почти не меняется —
распознающая способность та же, сдвинут только порог решения. Это кандидат
на следующие дни, но текущая версия оставлена честным baseline без
балансировки, чтобы было с чем сравнивать.

## Признаки по типам

Числовые и категориальные признаки заданы явно в `src/churn_fastapi/config.py`
(`NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`) — эндпоинт `GET /dataset/split-info`
покажет, что попало в каждую группу.

Пропуски обрабатываются внутри пайплайна (`SimpleImputer`): числовые — медианой,
категориальные — модой. Импутеры обучаются **только на train**, поэтому
тестовая выборка не участвует в подборе значений заполнения.

## Стек

- **FastAPI** — веб-фреймворк
- **scikit-learn** — обучение модели (LogisticRegression)
- **pandas** — работа с данными
- **joblib** — сохранение/загрузка модели
- **Pydantic** — валидация данных
- **uvicorn** — ASGI-сервер
