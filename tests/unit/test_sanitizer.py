# pyrefly: ignore [missing-import]
from app.services.sanitizer import Sanitizer


def test_map_offset_exclusive_end_identity():
    text = "22 Hai Bà Trưng"
    result = Sanitizer().sanitize(text)
    assert result.map_offset(len(result.offset_map)) == len(text)


def test_map_offset_exclusive_end_after_script_strip():
    text = "<script>alert('xss')</script> ship tới 45 Lê Lợi Q1"
    result = Sanitizer().sanitize(text)
    assert result.map_offset(len(result.offset_map)) == len(text)
