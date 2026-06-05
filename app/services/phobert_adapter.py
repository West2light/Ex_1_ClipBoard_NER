"""PhoBERT word segmentation adapter (M2).

PhoBERT was trained on Vietnamese text where multi-syllable words are
joined with underscores ("Hà_Nội", "Nguyễn_Văn_A").  This module:

1. Segments raw Vietnamese text via underthesea.word_tokenize.
2. Builds a character-level offset map from the segmented text back to
   the original raw text so that NER spans can be reconstructed correctly.

Only imported when MODEL_FAMILY=phobert.
"""

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_PHONE_PLACEHOLDER = "[PHONE]"
_PHONE_SENTINEL = "PH0NE99"


def _try_import_underthesea():
    try:
        import underthesea  # noqa: F401
        return True
    except ImportError:
        return False


_HAS_UNDERTHESEA = _try_import_underthesea()


def _segment_text(raw_text: str) -> str:
    """Return word-segmented text with underscore-joined tokens."""
    if not _HAS_UNDERTHESEA:
        logger.warning(
            "underthesea not installed — PhoBERT segmentation disabled, "
            "falling back to raw text."
        )
        return raw_text

    from underthesea import word_tokenize  # type: ignore

    protected_text = raw_text.replace(_PHONE_PLACEHOLDER, _PHONE_SENTINEL)
    segmented: str = word_tokenize(protected_text, format="text")
    return segmented.replace(_PHONE_SENTINEL, _PHONE_PLACEHOLDER)


def _build_offset_map(raw_text: str, segmented_text: str) -> List[int]:
    """Map each character position in *segmented_text* to a position in *raw_text*.

    Strategy:
    - Walk both strings in parallel.
    - When a '_' separator appears in segmented that wasn't in raw, record the
      next raw position (the continuation position).
    - This handles the case "Hà_Nội" in segmented vs "Hà Nội" in raw.

    Returns a list `offset_map` where ``offset_map[i]`` is the raw-text
    character index corresponding to segmented-text character index ``i``.
    If the map cannot be built (lengths diverge unexpectedly), falls back to
    identity mapping.
    """
    seg_len = len(segmented_text)
    raw_len = len(raw_text)

    offset_map: List[int] = []
    r = 0  # current search position in raw_text

    for ch_seg in segmented_text:
        if r >= raw_len:
            offset_map.append(raw_len)
            continue

        if ch_seg == "_":
            while r < raw_len and raw_text[r].isspace():
                r += 1
            offset_map.append(max(0, r - 1))
            continue

        if ch_seg.isspace():
            if raw_text[r].isspace():
                offset_map.append(r)
                r += 1
            else:
                # underthesea may insert spaces around punctuation.
                offset_map.append(r)
            continue

        match = raw_text.find(ch_seg, r)
        if match < 0:
            offset_map.append(r)
            r += 1
        else:
            offset_map.append(match)
            r = match + 1

    return offset_map


def segment_and_map(raw_text: str) -> Tuple[str, List[int]]:
    """Segment raw Vietnamese text for PhoBERT and build offset map.

    Returns:
        (segmented_text, offset_map)

        offset_map[i] is the raw-text char index for segmented char i.
        Use ``offset_map[span_start]`` and ``offset_map[span_end - 1] + 1``
        to convert a segmented span back to a raw span.
    """
    segmented = _segment_text(raw_text)
    offset_map = _build_offset_map(raw_text, segmented)
    return segmented, offset_map


def segmented_span_to_raw(
    seg_start: int,
    seg_end: int,
    offset_map: List[int],
    raw_len: int,
) -> Tuple[int, int]:
    """Convert a [seg_start, seg_end) span to raw text coordinates.

    Returns (raw_start, raw_end) clamped to [0, raw_len].
    """
    if not offset_map or seg_start >= len(offset_map):
        return seg_start, seg_end

    raw_start = offset_map[seg_start]
    seg_end_idx = min(seg_end - 1, len(offset_map) - 1)
    raw_end = offset_map[seg_end_idx] + 1 if seg_end > 0 else raw_start

    raw_start = max(0, min(raw_start, raw_len))
    raw_end = max(raw_start, min(raw_end, raw_len))
    return raw_start, raw_end
