"""FastAPI application factory (M4 — main, v4).

Uses the modern lifespan context manager instead of deprecated on_event.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from app.core.config import settings
from app.core.logging import configure_logging
from app.core.errors import (
    AppError,
    InvalidInputError,
    app_error_handler,
    generic_error_handler,
)
from app.services.ner_service import NERService
from app.services.parser_service import ParserService
from app.api.routes import router

# Configure logging before anything else
configure_logging(settings.LOG_LEVEL)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the NER model at startup; nothing to clean up on shutdown."""
    model_path = settings.resolved_model_path
    logger.info(
        "Loading NER model — family=%s path=%s",
        settings.MODEL_FAMILY,
        model_path,
    )
    ner = NERService(model_path=model_path)
    if ner.is_ready():
        logger.info("NER model ready [OK]")
    else:
        logger.warning("NER model NOT ready — check model path: %s", model_path)

    app.state.parser = ParserService(ner_service=ner)
    yield
    # (shutdown cleanup would go here)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Clipboard Parsing API",
        version="4.0.0",
        description="Hybrid Vietnamese delivery-order NER parser (ONNX INT8 CPU)",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Exception handlers
    # ------------------------------------------------------------------

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        return await app_error_handler(request, exc)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        return await app_error_handler(request, InvalidInputError())

    @app.exception_handler(Exception)
    async def _generic(request: Request, exc: Exception):
        logger.exception("Unhandled exception in request")
        return await generic_error_handler(request, exc)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    app.include_router(router)

    return app


app = create_app()
