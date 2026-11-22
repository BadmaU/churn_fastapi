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

Можно выбрать тип модели и гиперпараметры:

```bash
curl -X POST http://localhost:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "hyperparameters": {"n_estimators": 100, "max_depth": 8}}'
```

`model_type`: `logreg` (по умолчанию) или `random_forest`.

### Схема признаков

```bash
curl http://localhost:8000/model/schema
```

Возвращает список признаков `churn` в порядке обучения, ожидаемые типы, группу (`numeric`/`categorical`) и ограничения.

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
| `POST` | `/model/train` | Обучить модель (`logreg` или `random_forest`), вернуть accuracy и f1 |
| `POST` | `/train` | Алиас `/model/train` |
| `GET` | `/model/status` | Обучена ли модель, тип и гиперпараметры, когда обучена, метрики |
| `GET` | `/model/metrics` | Метрики последнего обучения, история запусков и лучший запуск |
| `GET` | `/model/schema` | Список признаков churn, типы, группы и ограничения |
| `POST` | `/predict` | Предсказание для одного клиента |
| `POST` | `/predict/batch` | Пакетное предсказание |
| `GET` | `/metrics` | Метрики качества модели |
| `POST` | `/upload` | Загрузка CSV-датасета |
| `GET` | `/dataset` | Информация о текущем датасете |
| `GET` | `/dataset/info` | Размер датасета, признаки, распределение `churn` |
| `GET` | `/dataset/preview?limit=10` | Первые N строк датасета в JSON |
| `GET` | `/dataset/split-info` | Размеры train/test и распределение `churn` в выборках |

## Обработка ошибок

Все ошибки возвращаются в едином формате — без трассировок стека в ответе:

```json
{
  "code": "validation_error",
  "message": "Некорректные данные запроса",
  "details": [
    {
      "field": "monthly_fee",
      "message": "Input should be a valid number",
      "type": "float_parsing"
    }
  ],
  "detail": "monthly_fee: Input should be a valid number"
}
```

- `code` — машинночитаемый код: `validation_error`, `invalid_features`, `empty_dataset`,
  `model_not_trained`, `invalid_model_config`, `dataset_not_found`, `not_found`, `internal_error`;
- `message` — понятное описание ошибки;
- `details` — детали (список ошибок валидации или контекст);
- `detail` — дубль `message` для обратной совместимости со старыми клиентами.

### Пример ошибки `/model/train`

```bash
curl -X POST http://localhost:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "unknown"}'
```

```json
{
  "code": "invalid_model_config",
  "message": "Не удалось обучить модель: Неизвестный тип модели: unknown. Доступные типы: logreg, random_forest",
  "details": {"status_code": 400},
  "detail": "Не удалось обучить модель: ..."
}
```

Пустой датасет даёт `400` с кодом `empty_dataset`, отсутствие файла датасета — `404` с `dataset_not_found`.

### Пример ошибки `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_fee": "abc",
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

```json
{
  "code": "validation_error",
  "message": "Некорректные данные запроса",
  "details": [{"field": "monthly_fee", "message": "Input should be a valid number", "type": "float_parsing"}],
  "detail": "monthly_fee: Input should be a valid number"
}
```

Неверное количество признаков (недостающий или лишний) тоже даёт `422` с кодом
`validation_error` — набор полей должен совпадать с `GET /model/schema`. Если модель ещё
не обучена, ответ `400` с кодом `model_not_trained`.

## Метрики и история обучений

Каждый запуск `/model/train` добавляет запись в `models/training_history.json`:
время (UTC), тип модели, гиперпараметры и метрики. История переживает перезапуск
сервиса и хранит не более 100 последних запусков.

```bash
curl "http://localhost:8000/model/metrics?limit=5"
curl "http://localhost:8000/model/metrics?model_type=random_forest&limit=3"
```

Ответ:

```json
{
  "latest": {"trained_at": "...", "model_type": "random_forest", "hyperparameters": {}, "metrics": {"accuracy": 0.85, "f1": 0.84}},
  "history": [
    {"trained_at": "...", "model_type": "random_forest", "hyperparameters": {}, "metrics": {"accuracy": 0.85, "f1": 0.84}}
  ],
  "best": {"trained_at": "...", "model_type": "logreg", "hyperparameters": {}, "metrics": {"accuracy": 0.86, "f1": 0.86}},
  "total": 7
}
```

- `latest` — метрики последнего обучения;
- `history` — последние `limit` записей в хронологическом порядке (`1..100`, по умолчанию 10);
- `best` — лучший запуск по `f1` (при равенстве — по `accuracy`), считается по всей
  выборке с учётом фильтра;
- `total` — сколько всего записей подходит под фильтр.

`model_type` отбирает только нужные конфигурации, поэтому `GET /model/metrics`
можно использовать для сравнения `logreg` и `random_forest` с разными
гиперпараметрами: `best` показывает лучшую комбинацию, а `history` — динамику
качества. Старый `GET /metrics` по-прежнему возвращает только метрики текущей
модели.

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
│       ├── history.py         # История обучений в JSON
│       ├── errors.py          # APIError и коды ошибок
│       └── config.py          # Конфигурация и пути
├── models/                    # churn_model.joblib, model_metadata.json, training_history.json (gitignore)
└── tests/
    ├── conftest.py            # Фикстуры для тестов
    ├── test_api.py            # Тесты API
    ├── test_dataset.py        # Тесты загрузки и просмотра датасета
    ├── test_preprocessing.py  # Тесты предобработки и разбиения
    ├── test_model.py          # Тесты обучения модели
    ├── test_errors.py         # Тесты формата ошибок
    └── test_history.py        # Тесты истории обучений
```

## Хранение модели

Обученная модель не теряется при перезапуске сервиса:

| Файл | Содержимое |
|---|---|
| `models/churn_model.joblib` | сам `Pipeline` через `joblib.dump` |
| `models/model_metadata.json` | `trained_at`, метрики, список признаков, версия sklearn, размер датасета |
| `models/training_history.json` | история запусков: время, тип модели, гиперпараметры и метрики |

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
(`NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`) — эндпоинты `GET /model/schema` и
`GET /dataset/split-info` показывают группы и порядок признаков.

Предобработка объединена в единый `Pipeline`: `ColumnTransformer` применяет
`StandardScaler` к числовым и `OneHotEncoder` к категориальным признакам
(с `SimpleImputer` внутри пайплайна). Модель и вся цепочка предобработки
сохраняются/загружаются одним объектом через `joblib`, а подготовка
`FeatureVectorChurn` для предсказания использует `ALL_FEATURES` в порядке обучения.

## Стек

- **FastAPI** — веб-фреймворк
- **scikit-learn** — обучение модели (LogisticRegression)
- **pandas** — работа с данными
- **joblib** — сохранение/загрузка модели
- **Pydantic** — валидация данных
- **uvicorn** — ASGI-сервер
