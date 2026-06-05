import re
import unicodedata
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class PhoneCandidate:
    start: int
    end: int
    raw_text: str
    normalized: str
    accepted: bool
    selected: bool
    reject_reason: Optional[str]
    rank_reasons: List[str] = field(default_factory=list)

@dataclass
class PhoneExtractionResult:
    phone_number: Optional[str]
    phone_count: int
    candidates: List[PhoneCandidate]
    masked_text: str
    masked_phone_spans: List[tuple[int, int]] = field(default_factory=list)
    masked_to_sanitized_map: List[int] = field(default_factory=list)

    def map_masked_offset(self, masked_pos: int) -> int:
        """Map a masked-text character position back to sanitized-text space."""
        if not self.masked_to_sanitized_map:
            return masked_pos
        if masked_pos < 0:
            return 0
        if masked_pos >= len(self.masked_to_sanitized_map):
            return self.masked_to_sanitized_map[-1] + 1
        return self.masked_to_sanitized_map[masked_pos]

    def get_debug_metadata(self) -> List[Dict[str, Any]]:
        return [
            {
                "start": c.start,
                "end": c.end,
                "accepted": c.accepted,
                "selected": c.selected,
                "reject_reason": c.reject_reason,
                "rank_reasons": c.rank_reasons
            }
            for c in self.candidates
        ]

class PhoneExtractor:
    # Captures phones including +84, 84, 0, wrappers like ( ) - . spaces
    CANDIDATE_REGEX = re.compile(
        r'(?<!\d)\(?(?:\+?84|0)(?:[ \t\-\.\(\)]*\d){7,11}(?!\d)\)?'
    )
    NON_DIGIT_REGEX = re.compile(r'\D')

    def extract(self, text: str) -> PhoneExtractionResult:
        candidates = self._discover_candidates(text)
        candidates = self._validate_and_reject(candidates, text)
        candidates = self._rank_and_select(candidates, text)
        masked_text, masked_phone_spans, masked_to_sanitized_map = self._mask_text(
            text, candidates
        )
        
        selected_candidate = next((c for c in candidates if c.selected), None)
        phone_count = sum(1 for c in candidates if c.accepted)
        
        return PhoneExtractionResult(
            phone_number=selected_candidate.normalized if selected_candidate else None,
            phone_count=phone_count,
            candidates=candidates,
            masked_text=masked_text,
            masked_phone_spans=masked_phone_spans,
            masked_to_sanitized_map=masked_to_sanitized_map,
        )

    def _discover_candidates(self, text: str) -> List[PhoneCandidate]:
        candidates = []
        for match in self.CANDIDATE_REGEX.finditer(text):
            raw_text = match.group(0)
            normalized = self.NON_DIGIT_REGEX.sub('', raw_text)
            
            # Convert 84/ +84 to 0
            if normalized.startswith('84'):
                normalized = '0' + normalized[2:]
            
            candidates.append(PhoneCandidate(
                start=match.start(),
                end=match.end(),
                raw_text=raw_text,
                normalized=normalized,
                accepted=False,
                selected=False,
                reject_reason=None
            ))
        return candidates

    def _validate_and_reject(self, candidates: List[PhoneCandidate], text: str) -> List[PhoneCandidate]:
        text_lower = text.lower() # For context matching
        
        for c in candidates:
            # 1. Legacy prefix
            if c.normalized.startswith('012'):
                c.reject_reason = "legacy_prefix"
                continue
                
            # 2. Unsupported non-mobile (Assume valid mobiles start with 03, 05, 07, 08, 09)
            valid_prefixes = ('03', '05', '07', '08', '09')
            if not c.normalized.startswith(valid_prefixes):
                c.reject_reason = "unsupported_non_mobile"
                continue
                
            # 3. Invalid length
            if len(c.normalized) != 10:
                c.reject_reason = "invalid_length"
                continue

            # Context cues: house_number (nha so, so nha), order_id (ma don, ma van don)
            # Check up to 20 chars before the candidate
            prefix_context = text_lower[max(0, c.start - 20):c.start]
            prefix_context_norm = _strip_accents(prefix_context)
            
            if any(x in prefix_context_norm for x in ("nha so", "so nha")):
                c.reject_reason = "context_house_number"
                continue
            
            if any(x in prefix_context_norm for x in ("ma don", "ma van don")):
                c.reject_reason = "context_order_id"
                continue
                
            c.accepted = True

        return candidates

    def _rank_and_select(self, candidates: List[PhoneCandidate], text: str) -> List[PhoneCandidate]:
        valid_candidates = [c for c in candidates if c.accepted]
        if not valid_candidates:
            return candidates
            
        text_lower = text.lower()
        
        for c in valid_candidates:
            prefix_context = text_lower[max(0, c.start - 25):c.start]
            prefix_context_norm = _strip_accents(prefix_context)
            
            if "shop" in prefix_context_norm:
                c.rank_reasons.append("shop_context_downrank")
            else:
                c.rank_reasons.append("recipient_associated")
            
            positive_cues = ["sdt", "dt", "phone", "lien he"]
            if any(cue in prefix_context_norm for cue in positive_cues):
                c.rank_reasons.append("positive_phone_context")
                
        def rank_key(c: PhoneCandidate):
            # Prefer recipient_associated (not shop), then earlier appearance
            is_shop = "shop_context_downrank" in c.rank_reasons
            return (is_shop, c.start)

        sorted_valid = sorted(valid_candidates, key=rank_key)
        
        selected = sorted_valid[0]
        selected.selected = True
        
        # Add 'first_valid_candidate' to the selected one
        selected.rank_reasons.append("first_valid_candidate")
        
        return candidates

    def _mask_text(self, text: str, candidates: List[PhoneCandidate]) -> tuple[str, List[tuple[int, int]], List[int]]:
        accepted_candidates = sorted(
            [c for c in candidates if c.accepted],
            key=lambda c: c.start,
        )

        masked_chars: List[str] = []
        masked_phone_spans: List[tuple[int, int]] = []
        masked_to_sanitized_map: List[int] = []
        cursor = 0
        mask = "[PHONE]"
        mask_len = len(mask)

        for c in accepted_candidates:
            if c.start < cursor:
                continue

            while cursor < c.start:
                masked_chars.append(text[cursor])
                masked_to_sanitized_map.append(cursor)
                cursor += 1

            phone_len = max(1, c.end - c.start)
            masked_start = len(masked_chars)
            for i, ch in enumerate(mask):
                masked_chars.append(ch)
                if mask_len == 1 or phone_len == 1:
                    mapped = c.start
                else:
                    mapped = c.start + ((i * (phone_len - 1)) // (mask_len - 1))
                masked_to_sanitized_map.append(mapped)
            masked_phone_spans.append((masked_start, masked_start + mask_len))
            cursor = c.end

        while cursor < len(text):
            masked_chars.append(text[cursor])
            masked_to_sanitized_map.append(cursor)
            cursor += 1

        return "".join(masked_chars), masked_phone_spans, masked_to_sanitized_map


def _strip_accents(text: str) -> str:
    """Return a lower-level ASCII-ish context string for phone cue matching."""
    normalized = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return stripped.replace("đ", "d").replace("Đ", "D")
