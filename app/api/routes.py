"""API routes (M4) — POST /parse, GET /health, GET /ready."""

import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas import ParseRequest, ParseResponse, HealthResponse, ReadyResponse
from app.core.errors import InvalidInputError, ModelNotReadyError
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health() -> HealthResponse:
    """Liveness probe — returns 200 when process is running."""
    return HealthResponse(status="ok")


@router.get("/ready", tags=["Ops"])
async def ready(request: Request):
    """Readiness probe — returns 200 when NER model is loaded."""
    parser = request.app.state.parser
    if not parser.is_ready():
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "model_not_ready",
                    "message": "The parsing model is not ready.",
                },
                "debug": None,
            },
        )
    return ReadyResponse(
        status="ready",
        model_family=settings.MODEL_FAMILY,
        model_path=settings.resolved_model_path,
        use_phobert=settings.use_phobert,
    )


@router.post("/parse", response_model=ParseResponse, tags=["Parse"])
async def parse(payload: ParseRequest, request: Request):
    """Full hybrid parsing pipeline for Vietnamese delivery-order text."""
    parser = request.app.state.parser

    # Validate non-empty text (Pydantic only checks type, not emptiness)
    if not payload.text or not payload.text.strip():
        raise InvalidInputError()

    result = parser.parse(text=payload.text, debug=payload.debug)
    return ParseResponse(**result)
