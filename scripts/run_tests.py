"""
run_tests.py — Chạy toàn bộ test cases từ test_cases_parsing.jsonl
               và in báo cáo kết quả chi tiết.

Chạy:
    python scripts/run_tests.py [--url http://localhost:8000] [--layer all|parser_api|phone_extractor]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests

from app.services.phone_extractor import PhoneExtractor

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Màu terminal ANSI ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

JSONL_PATH = REPO_ROOT / "adds" / "test-case" / "test_cases_parsing.jsonl"
DEFAULT_LOG_DIR = REPO_ROOT / "adds" / "output"

SKIP_SCOPES = {"future"}          # R-05 geocode future scope
SKIP_LAYERS = {"performance", "privacy"}  # R-06 benchmark, R-09 privacy


def load_cases(layer_filter: str) -> list[dict]:
    cases = []
    with JSONL_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if case.get("scope") in SKIP_SCOPES:
                continue
            if case.get("layer") in SKIP_LAYERS:
                continue
            if layer_filter != "all" and case.get("layer") != layer_filter:
                continue
            cases.append(case)
    return cases


def build_text_factory(case: dict) -> str | None:
    """R-07: text_factory generator."""
    factory = case["input"].get("text_factory")
    if factory:
        return factory["character"] * factory["count"]
    return None


class Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _make_log_path(log_dir: Path, layer: str, run_id: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = run_id or layer
    base = log_dir / f"test_run_{stamp}_{suffix}.log"
    if not base.exists():
        return base

    idx = 2
    while True:
        candidate = log_dir / f"test_run_{stamp}_{suffix}_{idx}.log"
        if not candidate.exists():
            return candidate
        idx += 1


def run_api_case(base_url: str, case: dict) -> dict:
    """Gọi /parse với input từ test case, trả về dict kết quả."""
    inp = case["input"]

    text = inp.get("text")
    if text is None:
        text = build_text_factory(case)
    if text is None:
        return {"skip": True, "reason": "no_text_input"}

    payload = {"text": text, "debug": inp.get("debug", False)}
    exp = case["expected"]

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/parse",
            json=payload,
            timeout=10,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
    except requests.RequestException as e:
        return {"error": str(e), "passed": False}
    elapsed_ms = (time.perf_counter() - t0) * 1000

    actual_status = resp.status_code
    exp_status    = exp.get("http_status", 200)

    passed = actual_status == exp_status
    failures = []

    if not passed:
        failures.append(
            f"  HTTP {actual_status} ≠ expected {exp_status}"
        )

    try:
        body = resp.json()
    except Exception:
        body = {}

    # Kiểm tra error code
    exp_error = exp.get("error")
    if exp_error:
        actual_code = body.get("error", {}).get("code") if isinstance(body.get("error"), dict) else None
        if actual_code != exp_error.get("code"):
            failures.append(
                f"  error.code='{actual_code}' ≠ expected '{exp_error.get('code')}'"
            )
            passed = False

    # Kiểm tra response fields khi HTTP 200
    exp_resp = exp.get("response")
    if exp_resp and actual_status == 200:
        for field, exp_val in exp_resp.items():
            act_val = body.get(field)
            if act_val != exp_val:
                failures.append(
                    f"  response.{field}={act_val!r} ≠ expected {exp_val!r}"
                )
                passed = False

    # Kiểm tra forbidden_empty_strings (R-02)
    if exp.get("forbidden_empty_strings") and actual_status == 200:
        for k, v in body.items():
            if v == "":
                failures.append(f"  response.{k} is empty string (should be null)")
                passed = False

    # Kiểm tra forbidden_response_fields (R-01)
    for forbidden in exp.get("forbidden_response_fields", []):
        if forbidden in body:
            failures.append(f"  forbidden field '{forbidden}' present in response")
            passed = False

    return {
        "passed": passed and not failures,
        "failures": failures,
        "actual_status": actual_status,
        "elapsed_ms": elapsed_ms,
        "body": body,
    }


def run_phone_case(case: dict) -> dict:
    """
    layer=phone_extractor: chạy PhoneExtractor trực tiếp để kiểm tra
    regex, normalize, context filter, masking, and selection.
    """
    text = case["input"].get("text", "")
    exp  = case["expected"]

    extractor = PhoneExtractor()
    t0 = time.perf_counter()
    result = extractor.extract(text)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    failures = []
    passed = True

    exp_phone_count = exp.get("phone_count", 0)
    exp_phone_num   = exp.get("phone_number")
    exp_masked      = exp.get("masked_text")
    exp_candidates  = exp.get("candidates", [])

    if result.phone_count != exp_phone_count:
        failures.append(f"  phone_count={result.phone_count} ≠ expected {exp_phone_count}")
        passed = False

    if result.phone_number != exp_phone_num:
        failures.append(f"  phone_number={result.phone_number!r} ≠ expected {exp_phone_num!r}")
        passed = False

    if exp_masked and result.masked_text != exp_masked:
        failures.append(f"  masked_text={result.masked_text!r}\n    ≠ expected {exp_masked!r}")
        passed = False

    if len(result.candidates) != len(exp_candidates):
        failures.append(
            f"  candidates len={len(result.candidates)} ≠ expected {len(exp_candidates)}"
        )
        passed = False

    for i, (exp_c, act_c) in enumerate(zip(exp_candidates, result.candidates)):
        if exp_c.get("accepted") != act_c.accepted:
            failures.append(
                f"  candidate[{i}].accepted={act_c.accepted} ≠ {exp_c.get('accepted')}"
            )
            passed = False
        if exp_c.get("normalized") is not None and exp_c.get("normalized") != act_c.normalized:
            failures.append(
                f"  candidate[{i}].normalized={act_c.normalized!r} ≠ {exp_c.get('normalized')!r}"
            )
            passed = False
        if exp_c.get("reject_reason") != act_c.reject_reason:
            failures.append(
                f"  candidate[{i}].reject_reason={act_c.reject_reason!r} ≠ {exp_c.get('reject_reason')!r}"
            )
            passed = False
        if "selected" in exp_c and exp_c.get("selected") != act_c.selected:
            failures.append(
                f"  candidate[{i}].selected={act_c.selected} ≠ {exp_c.get('selected')}"
            )
            passed = False

    return {
        "passed": passed and not failures,
        "failures": failures,
        "actual_status": 200,
        "elapsed_ms": elapsed_ms,
        "body": {
            "phone_number": result.phone_number,
            "phone_count": result.phone_count,
            "masked_text": result.masked_text,
            "candidates": result.get_debug_metadata(),
        },
    }


def print_result(case: dict, result: dict, verbose: bool = False):
    cid   = case["id"]
    layer = case.get("layer", "?")
    tags  = ", ".join(case.get("tags", []))

    if result.get("skip"):
        icon = f"{YELLOW}SKIP{RESET}"
        print(f"  {icon}  [{cid}] {result.get('reason','')}")
        return

    if result.get("error"):
        print(f"  {RED}ERR {RESET}  [{cid}] {result['error']}")
        return

    if result["passed"]:
        ms = result["elapsed_ms"]
        print(f"  {GREEN}PASS{RESET}  [{cid}] {DIM}{tags}{RESET}  {DIM}{ms:.0f}ms{RESET}")
    else:
        print(f"  {RED}FAIL{RESET}  [{cid}] {DIM}{tags}{RESET}")
        for msg in result.get("failures", []):
            print(f"{RED}{msg}{RESET}")
        if verbose:
            print(f"  {DIM}body: {json.dumps(result.get('body'), ensure_ascii=False)}{RESET}")


def main():
    parser = argparse.ArgumentParser(description="Clipboard NER API test runner")
    parser.add_argument("--url",     default="http://localhost:8000", help="Base URL của server")
    parser.add_argument("--layer",   default="all",
                        help="Lọc theo layer: all | parser_api | phone_extractor")
    parser.add_argument("--verbose", action="store_true", help="In body chi tiết khi FAIL")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Thư mục ghi log terminal")
    parser.add_argument("--run-id", default=None, help="Suffix log file, mặc định dùng layer")
    parser.add_argument("--no-log", action="store_true", help="Tắt ghi log terminal ra file")
    args = parser.parse_args()

    log_file = None
    log_path = None
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    try:
        if not args.no_log:
            log_dir = Path(args.log_dir)
            if not log_dir.is_absolute():
                log_dir = REPO_ROOT / log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = _make_log_path(log_dir, args.layer, args.run_id)
            log_file = log_path.open("w", encoding="utf-8", newline="\n")
            sys.stdout = Tee(orig_stdout, log_file)
            sys.stderr = Tee(orig_stderr, log_file)

        cases = load_cases(args.layer)
        print(f"\n{BOLD}{CYAN}═══ Clipboard NER Test Runner ═══{RESET}")
        print(f"Server : {args.url}")
        print(f"Layer  : {args.layer}")
        print(f"Cases  : {len(cases)}")
        if log_path is not None:
            print(f"Log    : {log_path}")
        print()

        totals = {"pass": 0, "fail": 0, "skip": 0, "error": 0}
        fail_ids: list[str] = []

        # Nhóm theo layer
        groups: dict[str, list] = {}
        for c in cases:
            lyr = c.get("layer", "unknown")
            groups.setdefault(lyr, []).append(c)

        for layer_name, layer_cases in groups.items():
            print(f"{BOLD}── {layer_name} ({len(layer_cases)} cases) ──{RESET}")
            for case in layer_cases:
                lyr = case.get("layer")
                if lyr == "phone_extractor":
                    result = run_phone_case(case)
                else:
                    result = run_api_case(args.url, case)

                print_result(case, result, verbose=args.verbose)

                if result.get("skip"):
                    totals["skip"] += 1
                elif result.get("error"):
                    totals["error"] += 1
                    fail_ids.append(case["id"])
                elif result["passed"]:
                    totals["pass"] += 1
                else:
                    totals["fail"] += 1
                    fail_ids.append(case["id"])
            print()

        # Summary
        total = sum(totals.values())
        print(f"{BOLD}═══ Kết quả ═══{RESET}")
        print(f"  {GREEN}PASS {totals['pass']}{RESET}  "
              f"{RED}FAIL {totals['fail']}{RESET}  "
              f"{YELLOW}SKIP {totals['skip']}{RESET}  "
              f"ERR  {totals['error']}  "
              f"/ {total} tổng")
        if fail_ids:
            print(f"\n  {RED}Failed IDs: {', '.join(fail_ids)}{RESET}")

        sys.exit(0 if totals["fail"] == 0 and totals["error"] == 0 else 1)
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    main()
