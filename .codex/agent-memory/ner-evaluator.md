# NER Evaluator Memory

## Confirmed Facts

- The XLM-R model predicts `PER`, `ADDR`, and `NOTE`.
- Raw phone-only inputs can be mislabeled as address spans.

## Decisions

- Evaluate production behavior using phone-masked input.
- Compare raw-versus-masked predictions before accepting parser changes.

## Reusable Evidence

- `adds/output/cell_out_put_test_case.md`
- `adds/test-case/test_cases_parsing.md`

