"""API health/ready endpoint tests (M6)."""

import pytest
from unittest.mock import MagicMock
from httpx import AsyncClient, ASGITransport

from app.main import create_app
from app.services.ner_service import NERService
from app.services.parser_service import ParserService


def _app_with_readiness(model_ready: bool):
    """Create app with mocked NER pre-injected into app.state."""
    app = create_app()
    ner = MagicMock(spec=NERService)
    ner.is_ready.return_value = model_ready
    ner.extract_tokens.return_value = []
    app.state.parser = ParserService(ner_service=ner)
    return app


@pytest.mark.asyncio
async def test_health_always_200():
    """Health endpoint always returns 200 regardless of model state."""
    for ready in (True, False):
        app = _app_with_readiness(model_ready=ready)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200, f"Expected 200 for ready={ready}"
        assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_ready_200_when_model_ready():
    app = _app_with_readiness(model_ready=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_ready_503_when_model_not_ready():
    app = _app_with_readiness(model_ready=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "model_not_ready"
