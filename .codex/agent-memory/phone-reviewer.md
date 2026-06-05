# Phone Reviewer Memory

## Confirmed Facts

- The current phone regex covers common spaces, dashes, dots, `+84`, and `84`
  formats.
- Regex alone is insufficient for negative contexts such as `ma don` and
  `nha so`.

## Decisions

- Treat regex matches as candidates, then apply context filtering and ranking.
- Mask valid non-rejected phone candidates before NER.

## Reusable Checks

- Review PH-01 through PH-14 in `adds/test-case/test_cases_parsing.md`.
- 2026-06-04: The frozen M0 fixtures require rejected debug candidates for
  `PH-06` to `PH-11`. A mobile-only candidate regex like
  `(?<!\d)(\+?84|0)[\s\-\.\(\)]*[3-9](?:[\s\-\.\(\)]*\d){8}(?!\d)` cannot
  emit `PH-06`, `PH-07`, `PH-08`, or `PH-09`, so G1 must verify candidate
  breadth separately from validation and rejection.
