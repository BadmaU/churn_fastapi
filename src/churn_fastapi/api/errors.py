import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from churn_fastapi.core.errors import code_for_status
from churn_fastapi.schemas import ErrorResponse

logger = logging.getLogger("uvicorn.error.churn_fastapi.api")


def _error_response(
    code: str,
    message: str,
    status_code: int,
    details: dict | list | None = None,
    legacy_detail: str | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        code=code,
        message=message,
        details=details,
        detail=message if legacy_detail is None else legacy_detail,
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def _log_error(request: Request, code: str, message: str) -> None:
    logger.warning("Ошибка %s: %s %s — %s", code, request.method, request.url.path, message)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code = getattr(exc, "code", None) or code_for_status(exc.status_code)
        _log_error(request, code, str(exc.detail))
        return _error_response(
            code=code,
            message=str(exc.detail),
            status_code=exc.status_code,
            details={"status_code": exc.status_code},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = []
        for error in exc.errors():
            location = [str(part) for part in error.get("loc", ()) if part != "body"]
            details.append(
                {
                    "field": ".".join(location) or "body",
                    "message": error.get("msg", ""),
                    "type": error.get("type", ""),
                }
            )
        legacy_detail = "; ".join(f"{item['field']}: {item['message']}" for item in details)
        _log_error(request, "validation_error", "Некорректные данные запроса")
        return _error_response(
            code="validation_error",
            message="Некорректные данные запроса",
            status_code=422,
            details=details,
            legacy_detail=legacy_detail,
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        message = str(exc)
        if "Отсутствуют обязательные признаки" in message:
            code = "invalid_features"
        elif "пуст" in message.lower():
            code = "empty_dataset"
        else:
            code = "invalid_data"
        _log_error(request, code, message)
        return _error_response(
            code=code,
            message=message,
            status_code=400,
            details={"exception": type(exc).__name__},
        )

    @app.exception_handler(TypeError)
    async def type_error_handler(request: Request, exc: TypeError) -> JSONResponse:
        _log_error(request, "invalid_data", str(exc))
        return _error_response(
            code="invalid_data",
            message=str(exc),
            status_code=400,
            details={"exception": type(exc).__name__},
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        message = str(exc)
        not_trained = "не обучен" in message.lower()
        code = "model_not_trained" if not_trained else "internal_error"
        status_code = 400 if not_trained else 500
        _log_error(request, code, message)
        return _error_response(
            code=code,
            message=message if not_trained else "Внутренняя ошибка сервера",
            status_code=status_code,
            details={"exception": type(exc).__name__},
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(
        request: Request, exc: FileNotFoundError
    ) -> JSONResponse:
        _log_error(request, "not_found", str(exc) or "Файл не найден")
        return _error_response(
            code="not_found",
            message=str(exc) or "Файл не найден",
            status_code=404,
            details={"exception": type(exc).__name__},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Необработанная ошибка при обработке запроса", exc_info=exc)
        return _error_response(
            code="internal_error",
            message="Внутренняя ошибка сервера",
            status_code=500,
        )
