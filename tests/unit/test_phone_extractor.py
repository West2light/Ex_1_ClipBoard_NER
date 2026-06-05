# pyrefly: ignore [missing-import]
import pytest
from app.services.phone_extractor import PhoneExtractor

@pytest.fixture
def extractor():
    return PhoneExtractor()

def test_ph_01_spaced(extractor):
    res = extractor.extract("Nguyễn Văn A 0912 345 678")
    assert res.phone_number == "0912345678"
    assert res.phone_count == 1
    assert res.masked_text == "Nguyễn Văn A [PHONE]"
    assert res.candidates[0].accepted

def test_ph_02_dashed(extractor):
    res = extractor.extract("0912-345-678")
    assert res.phone_number == "0912345678"
    assert res.phone_count == 1
    assert res.masked_text == "[PHONE]"

def test_ph_03_plus84(extractor):
    res = extractor.extract("+84912345678")
    assert res.phone_number == "0912345678"
    assert res.phone_count == 1
    assert res.masked_text == "[PHONE]"

def test_ph_04_bare84(extractor):
    res = extractor.extract("84912345678")
    assert res.phone_number == "0912345678"
    assert res.phone_count == 1
    assert res.masked_text == "[PHONE]"

def test_ph_05_wrapped(extractor):
    res = extractor.extract("Gọi (0912) 345.678")
    assert res.phone_number == "0912345678"
    assert res.phone_count == 1
    assert res.masked_text == "Gọi [PHONE]"

def test_ph_06_short(extractor):
    res = extractor.extract("Số 091234567")
    assert res.phone_number is None
    assert res.phone_count == 0
    assert res.masked_text == "Số 091234567"
    assert res.candidates[0].reject_reason == "invalid_length"

def test_ph_07_long(extractor):
    res = extractor.extract("09123456789")
    assert res.phone_number is None
    assert res.phone_count == 0
    assert res.candidates[0].reject_reason == "invalid_length"

def test_ph_08_landline(extractor):
    res = extractor.extract("02438222222")
    assert res.phone_number is None
    assert res.phone_count == 0
    assert res.candidates[0].reject_reason == "unsupported_non_mobile"

def test_ph_09_legacy(extractor):
    res = extractor.extract("0123456789")
    assert res.phone_number is None
    assert res.phone_count == 0
    assert res.candidates[0].reject_reason == "legacy_prefix"

def test_ph_10_house_number(extractor):
    res = extractor.extract("nha so 0987654321")
    assert res.phone_number is None
    assert res.phone_count == 0
    assert res.candidates[0].reject_reason == "context_house_number"


def test_ph_10_house_number_accented(extractor):
    res = extractor.extract("nhà số 0987654321, giao buổi chiều")
    assert res.phone_number is None
    assert res.phone_count == 0
    assert res.candidates[0].reject_reason == "context_house_number"
    
def test_ph_11_order_id(extractor):
    res = extractor.extract("ma don 0912345678")
    assert res.phone_number is None
    assert res.phone_count == 0
    assert res.candidates[0].reject_reason == "context_order_id"


def test_ph_11_order_id_accented(extractor):
    res = extractor.extract("mã đơn: 0912345678")
    assert res.phone_number is None
    assert res.phone_count == 0
    assert res.candidates[0].reject_reason == "context_order_id"

def test_ph_12_recipient_and_shop(extractor):
    res = extractor.extract("Lan 0901234567, lien he shop 0987654321")
    assert res.phone_number == "0901234567"
    assert res.phone_count == 2
    assert res.masked_text == "Lan [PHONE], lien he shop [PHONE]"
    assert res.candidates[0].selected
    assert not res.candidates[1].selected
    assert "shop_context_downrank" in res.candidates[1].rank_reasons

def test_ph_13_two_valid(extractor):
    res = extractor.extract("0901234567 0987654321")
    assert res.phone_number == "0901234567"
    assert res.phone_count == 2
    assert res.masked_text == "[PHONE] [PHONE]"
    assert res.candidates[0].selected

def test_ph_14_no_phone(extractor):
    res = extractor.extract("Hello World")
    assert res.phone_number is None
    assert res.phone_count == 0
    assert len(res.candidates) == 0
    assert res.masked_text == "Hello World"


def test_ph_15_newline_phone_does_not_cross_line(extractor):
    text = "Nguyễn Thị Hoa\n0934567890\n15 Lý Thường Kiệt, P.14, Q.10"
    res = extractor.extract(text)
    assert res.phone_number == "0934567890"
    assert res.phone_count == 1
    assert res.masked_text == "Nguyễn Thị Hoa\n[PHONE]\n15 Lý Thường Kiệt, P.14, Q.10"
    assert res.masked_phone_spans == [(15, 22)]
    assert len(res.masked_to_sanitized_map) == len(res.masked_text)
