"""Unit tests for Resolver — first-detected address rule (E-05).

Covers:
  - Single ADDR candidate → returned as-is
  - Two ADDR candidates → first (index 0) wins, regardless of keywords
  - Empty addr_candidates → AddressNotFoundError raised
  - Blank text in first candidate → AddressNotFoundError raised
  - _resolve_name suppresses sender via _PICKUP_CUES (unchanged behaviour)
"""

import pytest
from unittest.mock import MagicMock

from app.core.errors import AddressNotFoundError
from app.services.postprocessor import CandidateSpan
from app.services.resolver import Resolver
from app.services.phone_extractor import PhoneExtractionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phone_result(phone_number=None):
    result = MagicMock(spec=PhoneExtractionResult)
    result.phone_number = phone_number
    return result


def _span(text: str, start: int = 0) -> CandidateSpan:
    return CandidateSpan(start=start, end=start + len(text), text=text)


# ---------------------------------------------------------------------------
# _resolve_address — first-detected rule
# ---------------------------------------------------------------------------

class TestResolveAddress:

    def setup_method(self):
        self.resolver = Resolver()
        self.phone = _make_phone_result()

    def test_single_candidate_returned(self):
        addr = [_span("45 Lê Lợi Q1 HCM", start=0)]
        result = self.resolver.resolve(self.phone, [], addr, [], "45 Lê Lợi Q1 HCM")
        assert result["address_raw"] == "45 Lê Lợi Q1 HCM"

    def test_two_candidates_first_wins(self):
        """E-05: 'lấy hàng ở 12 Lý Thái Tổ, giao tới 88 Nguyễn Du'
        First detected span (12 Lý Thái Tổ) must be selected regardless
        of 'giao tới' keyword appearing before the second span.
        """
        text = "lấy hàng ở 12 Lý Thái Tổ, giao tới 88 Nguyễn Du"
        addr = [
            _span("12 Lý Thái Tổ", start=11),   # index 0 — first detected
            _span("88 Nguyễn Du", start=37),      # index 1 — second detected
        ]
        result = self.resolver.resolve(self.phone, [], addr, [], text)
        assert result["address_raw"] == "12 Lý Thái Tổ"

    def test_two_candidates_order_respected(self):
        """Even when the 'better' address is second, first still wins."""
        text = "ship tới 22 Ngô Quyền, lấy hàng ở 55 Trần Phú"
        addr = [
            _span("22 Ngô Quyền", start=9),
            _span("55 Trần Phú", start=35),
        ]
        result = self.resolver.resolve(self.phone, [], addr, [], text)
        assert result["address_raw"] == "22 Ngô Quyền"

    def test_three_candidates_first_wins(self):
        addr = [
            _span("Địa chỉ A", start=0),
            _span("Địa chỉ B", start=15),
            _span("Địa chỉ C", start=30),
        ]
        result = self.resolver.resolve(self.phone, [], addr, [], "Địa chỉ A, Địa chỉ B, Địa chỉ C")
        assert result["address_raw"] == "Địa chỉ A"

    def test_no_candidates_raises_address_not_found(self):
        with pytest.raises(AddressNotFoundError):
            self.resolver.resolve(self.phone, [], [], [], "không có địa chỉ")

    def test_blank_first_candidate_raises_address_not_found(self):
        """If first candidate has empty text, AddressNotFoundError must be raised."""
        addr = [_span("   ", start=0)]  # whitespace-only span
        with pytest.raises(AddressNotFoundError):
            self.resolver.resolve(self.phone, [], addr, [], "   ")

    def test_whitespace_stripped_from_result(self):
        addr = [_span("  45 Lê Lợi Q1  ", start=0)]
        result = self.resolver.resolve(self.phone, [], addr, [], "  45 Lê Lợi Q1  ")
        assert result["address_raw"] == "45 Lê Lợi Q1"


# ---------------------------------------------------------------------------
# _resolve_name — _PICKUP_CUES suppression still works
# ---------------------------------------------------------------------------

class TestResolveNameSenderSuppression:
    """Verify that _PICKUP_CUES-based sender suppression in _resolve_name
    is not broken by the address scoring change.
    """

    def setup_method(self):
        self.resolver = Resolver()
        self.phone = _make_phone_result()
        self.addr = [_span("22 Ngô Quyền", start=30)]

    def test_name_after_pickup_cue_suppressed(self):
        """PER span that follows a pickup-cue context should be suppressed."""
        text = "lấy hàng ở Minh Tú, giao tới 22 Ngô Quyền"
        # Simulate a PER span for "Minh Tú" right after "lấy hàng ở"
        per = [_span("Minh Tú", start=11)]
        result = self.resolver.resolve(self.phone, per, self.addr, [], text)
        assert result["recipient_name"] is None

    def test_normal_name_not_suppressed(self):
        """PER span not preceded by a pickup cue should pass through."""
        text = "Nguyễn Văn A, giao tới 22 Ngô Quyền"
        per = [_span("Nguyễn Văn A", start=0)]
        result = self.resolver.resolve(self.phone, per, self.addr, [], text)
        assert result["recipient_name"] == "Nguyễn Văn A"


# ---------------------------------------------------------------------------
# Phone resolution (unchanged)
# ---------------------------------------------------------------------------

class TestResolvePhone:

    def setup_method(self):
        self.resolver = Resolver()
        self.addr = [_span("45 Lê Lợi Q1", start=0)]

    def test_valid_phone_passed_through(self):
        phone = _make_phone_result("0912345678")
        result = self.resolver.resolve(phone, [], self.addr, [], "45 Lê Lợi Q1")
        assert result["phone_number"] == "0912345678"

    def test_null_phone_passed_through(self):
        phone = _make_phone_result(None)
        result = self.resolver.resolve(phone, [], self.addr, [], "45 Lê Lợi Q1")
        assert result["phone_number"] is None
