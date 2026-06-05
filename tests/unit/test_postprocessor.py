# pyrefly: ignore [missing-import]
import pytest
from app.services.postprocessor import PostProcessor, CandidateSpan
from app.services.ner_service import TokenRow

@pytest.fixture
def postprocessor():
    return PostProcessor()

def dummy_map_offset(idx: int) -> int:
    return idx

def test_assemble_bio(postprocessor):
    rows = [
        TokenRow(0, 6, "Nguyễn", "B-PER"),
        TokenRow(7, 10, "Văn", "I-PER"),
        TokenRow(11, 12, "A", "I-PER"),
        TokenRow(12, 13, " ", "O"),
        TokenRow(13, 15, "Số", "B-ADDR"),
        TokenRow(16, 17, "1", "I-ADDR")
    ]
    
    res = postprocessor.process(
        token_rows=rows,
        raw_text="Nguyễn Văn A Số 1",
        map_offset=dummy_map_offset,
        phone_mask_spans=[],
        script_style_spans=[]
    )
    
    assert len(res["per_candidates"]) == 1
    assert res["per_candidates"][0].text == "Nguyễn Văn A"
    
    assert len(res["addr_candidates"]) == 1
    assert res["addr_candidates"][0].text == "Số 1"

def test_phone_overlap_drop(postprocessor):
    rows = [
        TokenRow(0, 7, "[PHONE]", "B-ADDR")
    ]
    res = postprocessor.process(
        token_rows=rows,
        raw_text="[PHONE]",
        map_offset=dummy_map_offset,
        phone_mask_spans=[(0, 7)],
        script_style_spans=[]
    )
    assert len(res["addr_candidates"]) == 0

def test_script_overlap_drop(postprocessor):
    rows = [
        TokenRow(0, 3, "Lan", "B-PER")
    ]
    res = postprocessor.process(
        token_rows=rows,
        raw_text="Lan",
        map_offset=dummy_map_offset,
        phone_mask_spans=[],
        script_style_spans=[(0, 3)]
    )
    assert len(res["per_candidates"]) == 0

def test_per_vocative_drop(postprocessor):
    rows = [
        TokenRow(0, 6, "Lan ơi", "B-PER")
    ]
    res = postprocessor.process(
        token_rows=rows,
        raw_text="Lan ơi",
        map_offset=dummy_map_offset,
        phone_mask_spans=[],
        script_style_spans=[]
    )
    assert len(res["per_candidates"]) == 0

def test_addr_evidence(postprocessor):
    # No evidence (orphan I-ADDR or random word)
    rows_no_evidence = [
        TokenRow(0, 4, "xanh", "I-ADDR")
    ]
    res = postprocessor.process(
        token_rows=rows_no_evidence,
        raw_text="xanh",
        map_offset=dummy_map_offset,
        phone_mask_spans=[],
        script_style_spans=[]
    )
    assert len(res["addr_candidates"]) == 0
    
    # With evidence in text
    rows_evidence = [
        TokenRow(0, 4, "xanh", "I-ADDR")
    ]
    res2 = postprocessor.process(
        token_rows=rows_evidence,
        raw_text="xanh",  # "xanh" doesn't have evidence
        map_offset=dummy_map_offset,
        phone_mask_spans=[],
        script_style_spans=[]
    )
    assert len(res2["addr_candidates"]) == 0
    
    rows_real_evidence = [
        TokenRow(0, 8, "nhà xanh", "I-ADDR")
    ]
    res3 = postprocessor.process(
        token_rows=rows_real_evidence,
        raw_text="nhà xanh",
        map_offset=dummy_map_offset,
        phone_mask_spans=[],
        script_style_spans=[]
    )
    assert len(res3["addr_candidates"]) == 1

def test_note_overlap_addr(postprocessor):
    rows = [
        TokenRow(0, 7, "Số 1 HN", "B-ADDR"),
        TokenRow(0, 7, "Số 1 HN", "B-NOTE") # overlapping exactly
    ]
    res = postprocessor.process(
        token_rows=rows,
        raw_text="Số 1 HN",
        map_offset=dummy_map_offset,
        phone_mask_spans=[],
        script_style_spans=[]
    )
    assert len(res["addr_candidates"]) == 1
    assert len(res["note_candidates"]) == 0
