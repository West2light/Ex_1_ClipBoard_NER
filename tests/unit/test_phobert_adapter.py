"""Unit tests for PhoBERT word segmentation adapter (M6)."""

import pytest
from app.services.phobert_adapter import (
    _build_offset_map,
    segment_and_map,
    segmented_span_to_raw,
)


# ---------------------------------------------------------------------------
# _build_offset_map tests
# ---------------------------------------------------------------------------


def test_identity_no_underscores():
    """When no underscores are introduced, offset map should be identity."""
    raw = "Hà Nội"
    seg = "Hà Nội"
    omap = _build_offset_map(raw, seg)
    # Each segmented char should map to the same position in raw
    assert omap == list(range(len(raw)))


def test_underscore_replaces_space():
    """Underscore in segmented should map to the position of the next raw char."""
    raw = "Hà Nội"
    seg = "Hà_Nội"
    omap = _build_offset_map(raw, seg)
    # 'H'->0, 'à'->1, '_'->2(space in raw), 'N'->2(skip space), 'ộ'->3, 'i'->4... wait
    # raw:  H à   N ộ i
    # idx:  0 1 2 3 4 5
    # seg:  H à _ N ộ i
    # idx:  0 1 2 3 4 5
    assert len(omap) == len(seg)
    # The underscore at seg[2] maps to raw[2] (space position)
    assert omap[2] == 2  # '_' -> space position


def test_span_reconstruction():
    """After segmentation, NER span on segmented text → correct raw span."""
    raw = "Hà Nội"
    seg = "Hà_Nội"
    omap = _build_offset_map(raw, seg)
    # If NER tags the whole segmented "Hà_Nội" as B-ADDR (0:6)
    raw_start, raw_end = segmented_span_to_raw(0, len(seg), omap, len(raw))
    assert raw_start == 0
    assert raw_end <= len(raw)


def test_empty_input():
    raw = ""
    seg = ""
    omap = _build_offset_map(raw, seg)
    assert omap == []


def test_segment_and_map_returns_tuple():
    """segment_and_map should always return (str, list)."""
    text = "Nguyễn Văn A giao hàng"
    seg, omap = segment_and_map(text)
    assert isinstance(seg, str)
    assert isinstance(omap, list)
    assert len(seg) > 0


def test_segment_and_map_no_crash_on_empty():
    seg, omap = segment_and_map("")
    assert seg == ""
    assert omap == []


def test_segmented_span_to_raw_empty_map():
    raw_start, raw_end = segmented_span_to_raw(0, 3, [], 10)
    assert raw_start == 0
    assert raw_end == 3


def test_segmented_span_to_raw_beyond_map():
    omap = [0, 1, 2, 3]
    raw_start, raw_end = segmented_span_to_raw(2, 10, omap, 4)
    # seg_end-1 = 9 which is beyond omap, so omap[-1] or clamped
    assert raw_start == 2
    assert raw_end <= 4
