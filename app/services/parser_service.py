"""Parser service — full pipeline orchestration (M4).

Wires together:
  Sanitizer → PhoneExtractor → NERService → PostProcessor → Resolver

Also handles PhoBERT word-segmentation when MODEL_FAMILY=phobert.
"""

import logging
import time
import uuid
from typing import Optional

from app.core.config import settings
from app.core.errors import (
    AddressNotFoundError,
    InvalidInputError,
    InputTooLongError,
    ModelNotReadyError,
    DebugNotAllowedError,
)
from app.services.sanitizer import Sanitizer
from app.services.phone_extractor import PhoneExtractor
from app.services.ner_service import NERService
from app.services.postprocessor import PostProcessor
from app.services.resolver import Resolver

logger = logging.getLogger(__name__)


class ParserService:
    """Orchestrates the full hybrid clipboard parsing pipeline."""

    def __init__(self, ner_service: NERService, use_phobert: Optional[bool] = None):
        self._ner = ner_service
        self._sanitizer = Sanitizer()
        self._phone_extractor = PhoneExtractor()
        self._postprocessor = PostProcessor()
        self._resolver = Resolver()

        # PhoBERT adapter — imported lazily
        self._phobert_adapter = None
        if use_phobert is None:
            use_phobert = settings.use_phobert
        if use_phobert:
            try:
                from app.services import phobert_adapter  # noqa: F401
                self._phobert_adapter = phobert_adapter
            except Exception as exc:
                logger.warning("PhoBERT adapter unavailable: %s", exc)

    def is_ready(self) -> bool:
        return self._ner.is_ready()

    def parse(self, text: str, debug: bool = False) -> dict:
        """Run the full parsing pipeline.

        Args:
            text:  Raw clipboard text.
            debug: If True, requires DEBUG_ENABLED server config.

        Returns:
            dict with keys: recipient_name, phone_number, address_raw, note, debug

        Raises:
            InvalidInputError, InputTooLongError, DebugNotAllowedError,
            ModelNotReadyError, AddressNotFoundError
        """
        t_start = time.monotonic()
        request_id = str(uuid.uuid4())

        # --- Input validation ---
        if not isinstance(text, str) or not text.strip():
            raise InvalidInputError()

        if len(text) > settings.MAX_INPUT_LENGTH:
            raise InputTooLongError()

        if debug and not settings.DEBUG_ENABLED:
            raise DebugNotAllowedError()

        # --- Model readiness ---
        if not self._ner.is_ready():
            raise ModelNotReadyError()

        # --- Sanitize ---
        t_san = time.monotonic()
        san_result = self._sanitizer.sanitize(text)
        sanitized = san_result.sanitized_text
        t_san = int((time.monotonic() - t_san) * 1000)

        # --- Phone extraction (on sanitized text) ---
        t_ph = time.monotonic()
        phone_result = self._phone_extractor.extract(sanitized)
        masked_text = phone_result.masked_text
        phone_mask_spans = phone_result.masked_phone_spans
        t_ph = int((time.monotonic() - t_ph) * 1000)

        # --- NER (on masked text, possibly segmented for PhoBERT) ---
        t_ner = time.monotonic()
        ner_input = masked_text
        offset_map_phobert = None

        if self._phobert_adapter is not None:
            ner_input, offset_map_phobert = self._phobert_adapter.segment_and_map(masked_text)

        token_rows = self._ner.extract_tokens(ner_input)
        t_ner = int((time.monotonic() - t_ner) * 1000)

        # --- Offset mapping function ---
        # Converts positions in masked_text (or phobert segmented) → raw text
        def map_to_raw(pos: int) -> int:
            # 1. If PhoBERT, first remap from segmented → masked/sanitized
            if offset_map_phobert is not None:
                if pos < 0:
                    pos = 0
                elif pos >= len(offset_map_phobert):
                    pos = len(masked_text)
                else:
                    pos = offset_map_phobert[pos]
                # else clamp to end
            # 2. Then remap from masked → sanitized → raw
            pos = phone_result.map_masked_offset(pos)
            return san_result.map_offset(pos)

        # --- Post-process ---
        t_pp = time.monotonic()
        candidates = self._postprocessor.process(
            token_rows=token_rows,
            raw_text=text,
            map_offset=map_to_raw,
            phone_mask_spans=phone_mask_spans,
            script_style_spans=san_result.script_style_spans,
        )
        if self._phobert_adapter is not None:
            self._promote_phobert_address_from_note(candidates)
        t_pp = int((time.monotonic() - t_pp) * 1000)

        # --- Resolve ---
        t_res = time.monotonic()
        resolved = self._resolver.resolve(
            phone_result=phone_result,
            per_candidates=candidates["per_candidates"],
            addr_candidates=candidates["addr_candidates"],
            note_candidates=candidates["note_candidates"],
            sanitized_text=text,  # use raw text for cue lookups (offsets already mapped)
        )
        t_res = int((time.monotonic() - t_res) * 1000)

        t_total = int((time.monotonic() - t_start) * 1000)

        # --- Build debug object (PII-safe) ---
        debug_obj = None
        if debug and settings.DEBUG_ENABLED:
            debug_obj = self._build_debug(
                request_id=request_id,
                text=text,
                san_result=san_result,
                phone_result=phone_result,
                candidates=candidates,
                resolved=resolved,
                latency={
                    "sanitize_ms": t_san,
                    "phone_ms": t_ph,
                    "ner_ms": t_ner,
                    "postprocess_ms": t_pp,
                    "resolve_ms": t_res,
                    "total_ms": t_total,
                },
            )

        return {**resolved, "debug": debug_obj}

    def _promote_phobert_address_from_note(self, candidates: dict) -> None:
        """PhoBERT fallback for artifacts that confuse address spans with NOTE."""
        if candidates["addr_candidates"]:
            return

        note_candidates = []
        for candidate in candidates["note_candidates"]:
            if self._postprocessor._has_address_evidence(candidate.text):
                candidates["addr_candidates"].append(candidate)
            else:
                note_candidates.append(candidate)
        candidates["note_candidates"] = note_candidates

    # ------------------------------------------------------------------
    # PII-safe debug builder
    # ------------------------------------------------------------------

    def _build_debug(
        self, request_id, text, san_result, phone_result, candidates, resolved, latency
    ) -> dict:
        """Build a PII-safe debug dict per api_contract_v1.md."""
        return {
            "request_id": request_id,
            "pipeline_version": "v4",
            "model_family": settings.MODEL_FAMILY,
            "input": {
                "character_count": len(text),
                "line_count": text.count("\n") + 1,
            },
            "sanitization": {
                "actions": san_result.actions,
                "removed_character_count": san_result.removed_count,
            },
            "phone": {
                "phone_count": phone_result.phone_count,
                "candidates": phone_result.get_debug_metadata(),
            },
            "ner": {
                "spans": [
                    {
                        "label": c.text and "ADDR" or "SPAN",  # label not stored in CandidateSpan
                        "start": c.start,
                        "end": c.end,
                    }
                    for group in candidates.values()
                    for c in group
                ],
            },
            "resolved": {
                "recipient_name_present": resolved.get("recipient_name") is not None,
                "phone_number_present": resolved.get("phone_number") is not None,
                "address_raw_present": bool(resolved.get("address_raw")),
                "note_present": resolved.get("note") is not None,
            },
            "latency_ms": latency,
        }
