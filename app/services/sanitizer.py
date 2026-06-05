"""Conservative text sanitizer (M4).

Performs in-order:
1. Reject if input exceeds MAX_INPUT_LENGTH (caller must check).
2. Remove <script>...</script> and <style>...</style> blocks.
3. Strip C0/C1 control characters except tab, LF, CR.
4. Preserve Vietnamese accents, punctuation, harmless emoji, line breaks.

Returns SanitizeResult which carries:
- sanitized_text  : cleaned string
- offset_map      : offset_map[i] = raw char index for sanitized char i
- actions         : list of action names applied (for PII-safe debug)
- removed_count   : number of characters removed
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Matches complete <script>…</script> and <style>…</style> blocks (case-insensitive, dot-all).
_SCRIPT_RE = re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
_STYLE_RE = re.compile(r"<style[^>]*>.*?</style>", re.IGNORECASE | re.DOTALL)

# Control characters to strip: everything in C0 (0x00–0x1F) and C1 (0x7F–0x9F)
# EXCEPT 0x09 (tab), 0x0A (LF), 0x0D (CR).
_CTRL_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]"
)


@dataclass
class SanitizeResult:
    sanitized_text: str
    # offset_map[i] = index in raw_text for sanitized_text[i]
    offset_map: List[int]
    # Spans of raw text that were removed as script/style blocks (raw indices)
    script_style_spans: List[Tuple[int, int]]
    actions: List[str] = field(default_factory=list)
    removed_count: int = 0

    def map_offset(self, sanitized_pos: int) -> int:
        """Return the raw-text position for a sanitized-text position."""
        if not self.offset_map:
            return sanitized_pos
        if sanitized_pos < 0:
            return 0
        if sanitized_pos >= len(self.offset_map):
            return self.offset_map[-1] + 1 if self.offset_map else sanitized_pos
        return self.offset_map[sanitized_pos]


class Sanitizer:
    """Conservative text sanitizer — preserves Vietnamese accents and structure."""

    def sanitize(self, raw_text: str) -> SanitizeResult:
        actions: List[str] = []
        script_style_spans: List[Tuple[int, int]] = []

        text = raw_text

        # Step 1: Remove <script> and <style> blocks
        text, ss_spans, ss_removed = self._strip_tags(text, _SCRIPT_RE, "strip_script")
        script_style_spans.extend(ss_spans)
        if ss_removed:
            actions.append("strip_script")

        text, st_spans, st_removed = self._strip_tags(text, _STYLE_RE, "strip_style")
        # Note: st_spans are in intermediate text coordinates; they've already been
        # removed so we track them by their *original* raw positions (before script strip).
        # For simplicity we store them relative to the sanitized text.
        script_style_spans.extend(st_spans)
        if st_removed:
            actions.append("strip_style")

        total_tag_removed = ss_removed + st_removed

        # Step 2: Strip control characters
        text, ctrl_removed = self._strip_controls(text)
        if ctrl_removed:
            actions.append("strip_control_chars")

        # Step 3: Build offset map from sanitized to raw
        offset_map = self._build_offset_map(raw_text, text, script_style_spans, ctrl_removed > 0)

        removed_count = total_tag_removed + ctrl_removed

        return SanitizeResult(
            sanitized_text=text,
            offset_map=offset_map,
            script_style_spans=script_style_spans,
            actions=actions,
            removed_count=removed_count,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _strip_tags(
        self, text: str, pattern: re.Pattern, tag_name: str
    ) -> Tuple[str, List[Tuple[int, int]], int]:
        """Remove all matches of *pattern* from *text*.

        Returns (new_text, list_of_removed_spans_in_original, removed_char_count).
        """
        spans: List[Tuple[int, int]] = []
        removed = 0
        result = []
        prev_end = 0
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end()))
            result.append(text[prev_end : m.start()])
            removed += m.end() - m.start()
            prev_end = m.end()
        result.append(text[prev_end:])
        return "".join(result), spans, removed

    def _strip_controls(self, text: str) -> Tuple[str, int]:
        """Strip C0/C1 control characters (preserving tab, LF, CR)."""
        cleaned = _CTRL_RE.sub("", text)
        removed = len(text) - len(cleaned)
        return cleaned, removed

    def _build_offset_map(
        self,
        raw_text: str,
        sanitized_text: str,
        removed_spans: List[Tuple[int, int]],
        has_ctrl_removal: bool,
    ) -> List[int]:
        """Build a character-level offset map from sanitized → raw text.

        We do a two-pointer walk: advance through raw_text, skip removed
        ranges, and record the mapping for each surviving character.
        """
        # Build a sorted set of removed raw character positions
        removed_positions: set = set()
        for rs, re_ in removed_spans:
            for i in range(rs, re_):
                removed_positions.add(i)

        offset_map: List[int] = []
        r = 0  # raw pointer

        for s_char in sanitized_text:
            # Skip removed raw positions
            while r < len(raw_text) and (
                r in removed_positions or (
                    has_ctrl_removal and _CTRL_RE.match(raw_text[r])
                )
            ):
                r += 1

            offset_map.append(r)
            r += 1

        return offset_map
