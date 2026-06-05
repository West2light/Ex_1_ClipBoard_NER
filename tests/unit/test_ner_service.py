"""Unit tests for ONNX-based NERService (M6).

Tests use the real ONNX model (xlmr) when available, or mock the
ONNX session for pure unit testing.
"""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.services.ner_service import NERService, TokenRow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.abspath("model/xlmr_ner_onnx_int8")
MODEL_AVAILABLE = os.path.exists(os.path.join(MODEL_PATH, "model_quantized.onnx"))
PHOBERT_MODEL_PATH = os.path.abspath("model/phobert_ner_onnx_int8")
PHOBERT_MODEL_AVAILABLE = os.path.exists(
    os.path.join(PHOBERT_MODEL_PATH, "model_quantized.onnx")
)


@pytest.fixture(scope="module")
def ner_service_real():
    """Load the real ONNX NER service (requires extracted model)."""
    return NERService(model_path=MODEL_PATH)


@pytest.fixture(scope="module")
def ner_service_phobert_real():
    """Load the real PhoBERT ONNX NER service when available."""
    return NERService(model_path=PHOBERT_MODEL_PATH)


@pytest.fixture()
def ner_service_mocked():
    """NERService with a mocked ONNX session for pure unit tests."""
    service = NERService.__new__(NERService)
    service.model_path = "fake/path"
    service.id2label = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ADDR",
        4: "I-ADDR",
        5: "B-NOTE",
        6: "I-NOTE",
    }

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": np.array([[0, 1, 2, 3, 4, 5]]),
        "attention_mask": np.array([[1, 1, 1, 1, 1, 1]]),
        "offset_mapping": np.array([[[0, 0], [0, 3], [4, 7], [8, 9], [9, 10], [0, 0]]]),
    }
    mock_tokenizer.convert_ids_to_tokens = lambda x: f"tok{x}"
    service.tokenizer = mock_tokenizer

    # Mock ONNX session — returns logits (1, 6, 7) where position 1=B-PER, 3=B-ADDR
    logits = np.zeros((1, 6, 7), dtype=np.float32)
    logits[0, 1, 1] = 10.0  # B-PER
    logits[0, 2, 2] = 10.0  # I-PER
    logits[0, 3, 3] = 10.0  # B-ADDR

    mock_session = MagicMock()
    mock_session.run.return_value = [logits]
    mock_session.get_inputs.return_value = [
        MagicMock(name="input_ids"),
        MagicMock(name="attention_mask"),
    ]
    service.session = mock_session

    return service


# ---------------------------------------------------------------------------
# Tests: is_ready
# ---------------------------------------------------------------------------


def test_is_ready_missing_model():
    service = NERService(model_path="nonexistent/path")
    assert service.is_ready() is False


def test_is_ready_mocked(ner_service_mocked):
    assert ner_service_mocked.is_ready() is True


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="ONNX model not extracted")
def test_is_ready_real(ner_service_real):
    assert ner_service_real.is_ready() is True


@pytest.mark.skipif(not PHOBERT_MODEL_AVAILABLE, reason="PhoBERT ONNX model not extracted")
def test_is_ready_phobert_real(ner_service_phobert_real):
    assert ner_service_phobert_real.is_ready() is True


# ---------------------------------------------------------------------------
# Tests: extract_tokens
# ---------------------------------------------------------------------------


def test_extract_tokens_empty_returns_empty(ner_service_mocked):
    rows = ner_service_mocked.extract_tokens("")
    assert rows == []


def test_extract_tokens_mocked_returns_token_rows(ner_service_mocked):
    rows = ner_service_mocked.extract_tokens("Văn A 45")
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, TokenRow)
        assert hasattr(row, "start")
        assert hasattr(row, "end")
        assert hasattr(row, "token")
        assert hasattr(row, "label")


def test_extract_tokens_mocked_skips_special_tokens(ner_service_mocked):
    """Tokens with offset [0, 0] should be skipped."""
    rows = ner_service_mocked.extract_tokens("Văn A 45")
    # offsets [0,0] at index 0 and 5 should be skipped; 4 rows remain
    assert len(rows) == 4


def test_extract_tokens_mocked_labels(ner_service_mocked):
    rows = ner_service_mocked.extract_tokens("Văn A 45")
    labels = [r.label for r in rows]
    assert "B-PER" in labels
    assert "B-ADDR" in labels


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="ONNX model not extracted")
def test_extract_tokens_real_basic(ner_service_real):
    text = "Nguyễn Văn A, 45 Lê Lợi Quận 1"
    rows = ner_service_real.extract_tokens(text)
    assert len(rows) > 0
    for row in rows:
        assert row.label in {
            "O", "B-PER", "I-PER", "B-ADDR", "I-ADDR", "B-NOTE", "I-NOTE"
        }


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="ONNX model not extracted")
def test_extract_tokens_real_detects_per(ner_service_real):
    rows = ner_service_real.extract_tokens("Nguyễn Văn A")
    labels = [r.label for r in rows]
    assert any("PER" in label for label in labels)


@pytest.mark.skipif(not PHOBERT_MODEL_AVAILABLE, reason="PhoBERT ONNX model not extracted")
def test_extract_tokens_phobert_real_no_offset_mapping_crash(ner_service_phobert_real):
    rows = ner_service_phobert_real.extract_tokens("Nguyễn Văn A giao tới Q.Bình Thạnh HCM")
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, TokenRow)
        assert 0 <= row.start <= row.end
