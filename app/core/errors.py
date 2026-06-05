"""Typed application errors (M3).

Maps every business and operational failure to an HTTP status and a
machine-readable code string per api_error_matrix.md.
"""

from fastapi import Request
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------


class AppError(Exception):
    """Base class for all typed application errors."""

    http_status: int = 500
    code: str = "internal_error"
    message: str = "An internal error occurred."


class InvalidInputError(AppError):
    http_status = 400
    code = "invalid_input"
    message = "Input text is required or malformed."


class InputTooLongError(AppError):
    http_status = 400
    code = "input_too_long"
    message = "Input text exceeds the maximum allowed length."


class DebugNotAllowedError(AppError):
    http_status = 403
    code = "debug_not_allowed"
    message = "Debug output is not enabled for this service."


class AddressNotFoundError(AppError):
    http_status = 422
    code = "address_not_found"
    message = "A delivery address could not be resolved."


class ModelNotReadyError(AppError):
    http_status = 503
    code = "model_not_ready"
    message = "The parsing model is not ready."


# ---------------------------------------------------------------------------
# FastAPI exception handlers
# ---------------------------------------------------------------------------


def _error_body(error: AppError, debug=None) -> dict:
    return {
        "error": {
            "code": error.code,
            "message": error.message,
        },
        "debug": debug,
    }


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.http_status,
        content=_error_body(exc),
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    internal = AppError()
    return JSONResponse(
        status_code=internal.http_status,
        content=_error_body(internal),
    )
