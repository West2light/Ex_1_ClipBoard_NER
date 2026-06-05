# API Architect Memory

## Confirmed Decisions

- Initial service stack: Python, FastAPI, Pydantic, PyTorch, Transformers.
- Initial required field: `address_raw`.
- Initial optional fields: recipient name, phone number, and note.

## Approved Pipeline

```text
phone candidates -> context filter -> masking -> NER -> resolver
```

## Scope Boundary

- Address normalization and decomposition are later phases.

## Stable Contract Guardrails

- Internal extractor artifacts such as `masked_text` may exist for pipeline
  handoff and fixture verification, but API debug and logs must remain PII-safe
  and must not expose raw, sanitized, normalized, or masked customer text.

## 2026-06-04

- M4 freeze: public routes are exactly `POST /parse`, `GET /health`, and
  `GET /ready`; `/health` is liveness-only, `/ready` reflects model-loaded
  readiness.
- Parse input max length is 5000 Unicode code points measured before
  sanitization; oversized input fails closed with `400 input_too_long`.
- FastAPI/Pydantic native validation output must be normalized to typed
  `400 invalid_input`; API v1 reserves `422` for `address_not_found`.
- Module boundary freeze: routes stay thin; `parser_service` orchestrates
  sanitizer -> phone extractor -> NER -> postprocessor -> resolver.
