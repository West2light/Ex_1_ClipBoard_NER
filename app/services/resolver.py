"""Field resolver (M4).

Converts post-processed NER candidates into the final API response fields
per resolver_contract.md.

Inputs:
  - phone_result:   PhoneExtractionResult
  - per_candidates: List[CandidateSpan]
  - addr_candidates: List[CandidateSpan]
  - note_candidates: List[CandidateSpan]
  - sanitized_text: str  (for role-cue context lookups)

Raises AddressNotFoundError when no valid address can be resolved.
"""

import logging
import re
from typing import List, Optional

from app.core.errors import AddressNotFoundError
from app.services.phone_extractor import PhoneExtractionResult
from app.services.postprocessor import CandidateSpan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role-cue vocabulary
# Note: _PICKUP_CUES / _find_cue are used only in _resolve_name() to
# suppress sender names. Address selection uses first-detected order.
# ---------------------------------------------------------------------------

_PICKUP_CUES = [
    "lấy hàng ở", "shop ở", "kho ở", "bên gửi", "người gửi",
    "lấy tại", "lấy ở",
]

# Context window (chars before span) for cue lookup
_CUE_WINDOW = 40


def _find_cue(span_start: int, text: str, cue_list: List[str]) -> Optional[str]:
    """Return the first matching cue found in the window before span_start."""
    window = text[max(0, span_start - _CUE_WINDOW) : span_start].lower()
    for cue in cue_list:
        if cue in window:
            return cue
    return None


class Resolver:
    """Resolve NER candidates into final API response fields."""

    def resolve(
        self,
        phone_result: PhoneExtractionResult,
        per_candidates: List[CandidateSpan],
        addr_candidates: List[CandidateSpan],
        note_candidates: List[CandidateSpan],
        sanitized_text: str,
    ) -> dict:
        """Return dict with keys: recipient_name, phone_number, address_raw, note.

        Raises AddressNotFoundError if no valid address can be resolved.
        """
        address_raw = self._resolve_address(addr_candidates, sanitized_text)
        recipient_name = self._resolve_name(per_candidates, sanitized_text)
        phone_number = self._resolve_phone(phone_result)
        note = self._resolve_note(note_candidates, address_raw)

        return {
            "recipient_name": recipient_name,
            "phone_number": phone_number,
            "address_raw": address_raw,
            "note": note,
        }

    # ------------------------------------------------------------------
    # Address
    # ------------------------------------------------------------------

    def _resolve_address(
        self, addr_candidates: List[CandidateSpan], text: str
    ) -> str:
        """Return the first-detected address span.

        Rule: when multiple ADDR spans exist, the one that appears first
        in the original text (lowest position / index 0 in NER output)
        is selected. No delivery/pickup cue scoring is applied.
        """
        if not addr_candidates:
            raise AddressNotFoundError()

        best = addr_candidates[0]
        address_raw = best.text.strip()

        if not address_raw:
            raise AddressNotFoundError()

        return address_raw

    # ------------------------------------------------------------------
    # Recipient name
    # ------------------------------------------------------------------

    def _resolve_name(
        self, per_candidates: List[CandidateSpan], text: str
    ) -> Optional[str]:
        if not per_candidates:
            return None

        for span in per_candidates:
            candidate_text = span.text.strip()
            if not candidate_text:
                continue
            # Sender/source suppression: look for pickup cues near the PER span
            if _find_cue(span.start, text, _PICKUP_CUES):
                continue
            return candidate_text

        return None

    # ------------------------------------------------------------------
    # Phone
    # ------------------------------------------------------------------

    def _resolve_phone(self, phone_result: PhoneExtractionResult) -> Optional[str]:
        return phone_result.phone_number  # already normalized or None

    # ------------------------------------------------------------------
    # Note
    # ------------------------------------------------------------------

    def _resolve_note(
        self, note_candidates: List[CandidateSpan], address_raw: str
    ) -> Optional[str]:
        if not note_candidates:
            return None

        for span in note_candidates:
            candidate_text = self._clean_note_text(span.text)
            if not candidate_text:
                continue
            # Do not return a note that is identical to or contained in address_raw
            if candidate_text in address_raw or address_raw in candidate_text:
                continue
            return candidate_text

        return None

    def _clean_note_text(self, text: str) -> str:
        """Prefer quoted note content and trim common note prefixes."""
        candidate = text.strip()
        if not candidate:
            return ""

        quoted = re.findall(r'"([^"]+)"', candidate)
        if quoted:
            return quoted[0].strip()

        single_quoted = re.findall(r"'([^']+)'", candidate)
        if single_quoted:
            return single_quoted[0].strip()

        lowered = candidate.lower()
        prefixes = (
            "nhớ ghi ",
            "ghi chú: ",
            "ghi chú ",
            "ghi: ",
            "note: ",
            "note ",
            "chú ý: ",
            "chú ý ",
        )
        for prefix in prefixes:
            if lowered.startswith(prefix):
                return candidate[len(prefix):].strip(" -:|,")

        return candidate
