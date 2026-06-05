# API Worker Memory

## Implementation Constraints

- Keep implementation work narrowly scoped to the assigned task.
- Do not modify generated datasets or model artifacts.
- Add focused tests for parser behavior changes.

## Approved Service Direction

- Use FastAPI for the first service version.
- Keep deterministic phone extraction separate from NER inference.
- 2026-06-04: M4 freeze is implementation-feasible with `/health` as liveness,
  `/ready` as model-loaded readiness, and `/parse` short-circuiting to
  `503 model_not_ready` when startup wiring is incomplete.
- 2026-06-04: Preferred API module layout is `app/main.py`, `app/api/routes.py`,
  `app/schemas.py`, `app/core/{config,errors,logging}.py`,
  `app/services/{parser_service,sanitizer,phone_extractor,ner_service,postprocessor,resolver}.py`.
