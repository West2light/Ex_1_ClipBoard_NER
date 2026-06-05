import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.ner_service import NERService
from app.services.phone_extractor import PhoneExtractor
from app.core.config import settings

ner = NERService(model_path=settings.resolved_model_path)
ph = PhoneExtractor()

raw = "Nguyễn Thị Hoa\n0934567890\n15 Lý Thường Kiệt, P.14, Q.10"
print(f"raw len={len(raw)}")

ph_result = ph.extract(raw)
print(f"phone candidates: {[(c.normalized, c.accepted, c.start, c.end) for c in ph_result.candidates]}")
print(f"masked_text: {repr(ph_result.masked_text)}")

masked = ph_result.masked_text
print(f"masked len={len(masked)}")

rows = ner.extract_tokens(masked)
labeled = [(r.token, r.label, r.start, r.end) for r in rows if r.label != "O"]
print(f"NER on masked:")
for tok, lbl, s, e in labeled:
    print(f"  {lbl:8s} [{s:3d}:{e:3d}] {tok!r}")

# Show overlap check
phone_mask_spans = [(c.start, c.end) for c in ph_result.candidates if c.accepted]
print(f"\nphone_mask_spans (sanitized coords): {phone_mask_spans}")
for tok, lbl, s, e in labeled:
    if lbl.endswith("ADDR"):
        for ps, pe in phone_mask_spans:
            overlap = max(s, ps) < min(e, pe)
            print(f"  _overlaps({s},{e}, {[(ps,pe)]}) = {overlap}  <-- {'BUG drops ADDR!' if overlap else 'OK'}")
