# Project Instructions

## Objective

Build a hybrid Clipboard Parsing API for Vietnamese delivery-order text.

The first delivery scope extracts:

- recipient name
- phone number
- raw address span
- delivery note

## Approved Pipeline

Use this processing order:

```text
raw text
-> validate and conservatively sanitize
-> generate phone candidates with regex
-> apply phone context filters and ranking
-> mask valid non-rejected phones as [PHONE]
-> run XLM-R NER for PER, ADDR, NOTE
-> post-process spans and resolve final fields
-> require address or return address_not_found
```

Do not ask the NER model to extract phone numbers.

## Current Evidence

- Model artifact: `model/ner_xlmr_clipboard.zip`
- NER labels: `PER`, `ADDR`, `NOTE`
- Main test cases: `adds/test-case/test_cases_parsing.md`
- Recorded raw-model output: `adds/output/cell_out_put_test_case.md`
- Latest architecture brainstorm:
  `adds/PLAN_HYBRID_CLIPBOARD_PARSING_API_SERVICE_V2_BRAINSTORM.md`

## Engineering Rules

- Use Python and FastAPI for the inference API unless an approved plan changes
  the service boundary.
- Keep raw Vietnamese accents and punctuation for NER.
- Treat `address_raw` as the required source-of-truth field in the initial
  scope.
- Return `null`, not empty strings, for missing optional fields.
- Never store API keys, secrets, raw customer text, or phone numbers in project
  memory files.
- Do not modify generated datasets or model artifacts unless explicitly asked.
- Add focused tests for every parser behavior change.
- Prefer deterministic rules for phone extraction and typed API errors.

## Verification

For phone behavior, verify against PH test cases in:

```text
adds/test-case/test_cases_parsing.md
```

For model behavior, compare raw and masked inference outputs.

For API behavior, verify:

```text
empty input -> HTTP 400 invalid_input
address missing -> HTTP 422 address_not_found
invalid phone with valid address -> HTTP 200 and phone_number null
```

## Subagent Coordination

Custom project agents live in `.codex/agents/`.

- `phone-reviewer`: phone regex, normalization, context filtering, masking
- `ner-evaluator`: model spans and raw-versus-masked evaluation
- `api-architect`: architecture and contracts
- `api-worker`: approved implementation work

Parallel reviewers should not edit application code. Use one implementation
worker at a time to avoid conflicts.

## Project Agent Memory

Codex native memories are global local state, not isolated per custom agent.
This project uses explicit per-agent note files under `.codex/agent-memory/`.

Each custom agent:

1. reads its own memory file at task start
2. may update only its own memory file
3. records only stable decisions, recurring pitfalls, and reusable commands
4. does not store secrets, raw PII, transient logs, or speculative findings

When a memory entry conflicts with this file or the user's latest request,
this file and the latest request win.

