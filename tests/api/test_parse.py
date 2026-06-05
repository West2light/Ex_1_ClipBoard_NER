"""API contract tests for POST /parse (M6).

Uses httpx AsyncClient + ASGITransport. Since ASGITransport does NOT
trigger the app lifespan, we inject app.state.parser directly after
creating the app but before sending requests.
"""

import pytest
from unittest.mock import MagicMock
from httpx import AsyncClient, ASGITransport

from app.main import create_app
from app.services.ner_service import NERService, TokenRow
from app.services.parser_service import ParserService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_ner(token_rows=None, side_effect=None):
    mock_ner = MagicMock(spec=NERService)
    mock_ner.is_ready.return_value = True
    if side_effect is not None:
        mock_ner.extract_tokens.side_effect = side_effect
    else:
        mock_ner.extract_tokens.return_value = token_rows or []
    return mock_ner


def _app_with_parser(token_rows=None, side_effect=None):
    """Return a fully wired FastAPI app with mocked NER pre-injected."""
    app = create_app()
    ner = _make_mock_ner(token_rows, side_effect=side_effect)
    app.state.parser = ParserService(ner_service=ner, use_phobert=False)
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_returns_200():
    app = _app_with_parser()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_ready_when_model_ready():
    app = _app_with_parser()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ready")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_empty_input_returns_400():
    app = _app_with_parser()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/parse", json={"text": ""})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"


@pytest.mark.asyncio
async def test_whitespace_only_returns_400():
    app = _app_with_parser()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/parse", json={"text": "   "})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"


@pytest.mark.asyncio
async def test_missing_text_field_returns_400():
    app = _app_with_parser()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/parse", json={})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"


@pytest.mark.asyncio
async def test_unknown_field_returns_400():
    app = _app_with_parser()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/parse", json={"text": "some text", "unknown_field": 1})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"


@pytest.mark.asyncio
async def test_no_address_returns_422():
    """Text with no NER-resolved address → 422 address_not_found."""
    app = _app_with_parser([])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/parse", json={"text": "0912345678"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "address_not_found"


@pytest.mark.asyncio
async def test_valid_address_returns_200():
    """NER returns B-ADDR → 200 with non-empty address_raw."""
    rows = [TokenRow(start=0, end=13, token="▁45_Lê_Lợi_Q1", label="B-ADDR")]
    app = _app_with_parser(rows)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/parse", json={"text": "45 Lê Lợi Q1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["address_raw"]


@pytest.mark.asyncio
async def test_invalid_phone_with_valid_address_phone_null():
    """Legacy 012-prefix phone must not block 200; phone_number must be null."""
    text = "Nguyễn Văn A, 0123456789, 22 Hai Bà Trưng"
    addr_start = text.index("22 Hai")
    rows = [TokenRow(start=addr_start, end=addr_start + 15, token="▁addr", label="B-ADDR")]
    app = _app_with_parser(rows)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/parse", json={"text": text})
    assert resp.status_code == 200
    assert resp.json()["phone_number"] is None


@pytest.mark.asyncio
async def test_input_too_long_returns_400():
    """Input > 5000 chars → 400 input_too_long."""
    app = _app_with_parser()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/parse", json={"text": "a" * 5001})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "input_too_long"


@pytest.mark.asyncio
async def test_response_has_all_fields():
    """Successful response must always include all four business fields."""
    rows = [TokenRow(start=0, end=13, token="▁45_Lê_Lợi_Q1", label="B-ADDR")]
    app = _app_with_parser(rows)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/parse", json={"text": "45 Lê Lợi Q1"})
    assert resp.status_code == 200
    data = resp.json()
    for field in ("recipient_name", "phone_number", "address_raw", "note"):
        assert field in data


@pytest.mark.asyncio
async def test_address_after_phone_uses_masked_offset_map():
    """Address after a masked phone must still map back to the full raw span."""
    text = "Nguyễn Văn A, 0912345678, 45 Lê Lợi Q1 HCM"
    expected_address = "45 Lê Lợi Q1 HCM"

    def _extract_tokens(ner_input: str):
        start = ner_input.index(expected_address)
        return [TokenRow(start=start, end=start + len(expected_address), token="▁addr", label="B-ADDR")]

    app = _app_with_parser(side_effect=_extract_tokens)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/parse", json={"text": text})

    assert resp.status_code == 200
    data = resp.json()
    assert data["address_raw"] == expected_address


@pytest.mark.asyncio
async def test_phobert_promotes_address_like_note_when_addr_missing():
    """PhoBERT may label address spans as NOTE; parser should still resolve address."""
    text = "Nguyễn Văn A, 0912345678, 45 Lê Lợi Q1 HCM"
    expected_address = "45 Lê Lợi Q1 HCM"
    segmented_address = "45 Lê_Lợi_Q1 HCM"

    def _extract_tokens(ner_input: str):
        start = ner_input.index(segmented_address)
        return [
            TokenRow(
                start=start,
                end=start + len(segmented_address),
                token="addr",
                label="B-NOTE",
            )
        ]

    app = create_app()
    ner = _make_mock_ner(side_effect=_extract_tokens)
    app.state.parser = ParserService(ner_service=ner, use_phobert=True)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/parse", json={"text": text})

    assert resp.status_code == 200
    data = resp.json()
    assert data["address_raw"] == expected_address
    assert data["note"] is None


@pytest.mark.asyncio
async def test_two_addresses_returns_first_detected():
    """E-05: When NER detects two address spans, the first one in text order
    must be returned regardless of delivery/pickup keyword context.

    Input : lấy hàng ở 12 Lý Thái Tổ, giao tới 88 Nguyễn Du
    Expected address_raw: "12 Lý Thái Tổ"  (first detected)
    """
    text = "lấy hàng ở 12 Lý Thái Tổ, giao tới 88 Nguyễn Du"
    first_addr = "12 Lý Thái Tổ"
    second_addr = "88 Nguyễn Du"

    def _extract_tokens(ner_input: str):
        # Return both ADDR spans in text order (first → second)
        s1 = ner_input.index(first_addr)
        s2 = ner_input.index(second_addr)
        return [
            TokenRow(start=s1, end=s1 + len(first_addr), token="▁addr1", label="B-ADDR"),
            TokenRow(start=s2, end=s2 + len(second_addr), token="▁addr2", label="B-ADDR"),
        ]

    app = _app_with_parser(side_effect=_extract_tokens)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/parse", json={"text": text})

    assert resp.status_code == 200
    data = resp.json()
    assert data["address_raw"] == first_addr, (
        f"Expected first-detected '{first_addr}', got '{data['address_raw']}'"
    )
