"""
view_outputs.py — Gửi từng test case lên API và in output thực tế ra màn hình.
Mục đích: quan sát response thô, KHÔNG so sánh với expected.

Chạy:
    python scripts/view_outputs.py [--url http://localhost:8000] [--ids P-01 PH-03]
                                   [--layer parser_api|phone_extractor|all]
                                   [--debug]
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ── UTF-8 output Windows ──────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── ANSI colors ───────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUTS_PATH = REPO_ROOT / "adds" / "test-case" / "test_inputs.json"
DEFAULT_LOG_DIR = REPO_ROOT / "adds" / "output"

SKIP_SCOPES = {"future"}
SKIP_LAYERS = {"performance", "privacy"}


# ── Load test inputs ──────────────────────────────────────────────────────────

def load_inputs(layer_filter: str, id_filter: list[str]) -> list[dict]:
    with INPUTS_PATH.open(encoding="utf-8") as f:
        cases = json.load(f)

    result = []
    for case in cases:
        if layer_filter != "all" and case.get("layer") != layer_filter:
            continue
        if id_filter and case["id"] not in id_filter:
            continue
        # R-07: text = null → generate factory text
        if case.get("text") is None:
            note = case.get("note", "")
            if "5001" in note:
                case = dict(case)
                case["text"] = "a" * 5001
                case["_generated"] = True
        result.append(case)
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_LOG_REPLACEMENTS = str.maketrans(
    {
        "═": "=",
        "─": "-",
        "↵": "\\n",
        "✗": "X",
    }
)


def _clean_log_text(data: str) -> str:
    """Strip terminal styling so persisted logs are readable plain text."""
    return _ANSI_RE.sub("", data).translate(_LOG_REPLACEMENTS)


class Tee:
    def __init__(self, console_stream, log_stream):
        self._console_stream = console_stream
        self._log_stream = log_stream

    def write(self, data):
        self._console_stream.write(data)
        self._console_stream.flush()
        self._log_stream.write(_clean_log_text(data))
        self._log_stream.flush()

    def flush(self):
        self._console_stream.flush()
        self._log_stream.flush()


def _make_log_path(log_dir: Path, layer: str, run_id: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = run_id or layer
    base = log_dir / f"view_outputs_{stamp}_{suffix}.log"
    if not base.exists():
        return base

    idx = 2
    while True:
        candidate = log_dir / f"view_outputs_{stamp}_{suffix}_{idx}.log"
        if not candidate.exists():
            return candidate
        idx += 1


def truncate(s: str, max_len: int = 120) -> str:
    if len(s) > max_len:
        return s[:max_len] + f"  {DIM}[...+{len(s)-max_len} chars]{RESET}"
    return s


def status_color(code: int) -> str:
    if code == 200:
        return GREEN
    elif code in (400, 422):
        return YELLOW
    elif code >= 500:
        return RED
    return CYAN


def print_separator(char: str = "─", width: int = 72):
    print(f"{DIM}{char * width}{RESET}")


def print_header(case: dict):
    tag_str = "  ".join(f"#{t}" for t in case.get("tags", []))
    gen_flag = f"  {YELLOW}[generated]{RESET}" if case.get("_generated") else ""
    print()
    print_separator("═")
    print(
        f"{BOLD}{CYAN}[{case['id']}]{RESET}  "
        f"{DIM}layer={case.get('layer','?')}{RESET}"
        f"{gen_flag}"
    )
    if tag_str:
        print(f"  {DIM}{tag_str}{RESET}")
    print_separator()


def print_input(text: str):
    display = text.replace("\n", "↵ ")
    print(f"  {BOLD}INPUT :{RESET} {YELLOW}{truncate(display)}{RESET}")


def print_response(resp: requests.Response, elapsed_ms: float):
    color = status_color(resp.status_code)
    print(f"  {BOLD}STATUS:{RESET} {color}{resp.status_code}{RESET}  {DIM}({elapsed_ms:.0f} ms){RESET}")

    try:
        body = resp.json()
        pretty = json.dumps(body, ensure_ascii=False, indent=4)
        # indent & colorize keys
        lines = pretty.splitlines()
        for line in lines:
            # highlight keys
            stripped = line.lstrip()
            indent = " " * (len(line) - len(stripped))
            if stripped.startswith('"') and ":" in stripped:
                key_end = stripped.index(":") + 1
                key_part = stripped[:key_end]
                val_part = stripped[key_end:]
                val_color = RED if "null" in val_part else GREEN if "true" in val_part or "false" in val_part else RESET
                print(f"  {indent}{CYAN}{key_part}{RESET}{val_color}{val_part}{RESET}")
            else:
                print(f"  {indent}{stripped}")
    except Exception:
        raw = resp.text[:500]
        print(f"  {RED}[non-JSON response]{RESET}  {raw}")


# ── Main runner ───────────────────────────────────────────────────────────────

def run(
    url: str,
    layer_filter: str,
    id_filter: list[str],
    debug: bool,
    log_path: Path | None = None,
):
    cases = load_inputs(layer_filter, id_filter)
    if not cases:
        print(f"{RED}Không tìm thấy test case nào phù hợp.{RESET}")
        sys.exit(1)

    endpoint = url.rstrip("/") + "/parse"
    print(f"\n{BOLD}API endpoint:{RESET} {CYAN}{endpoint}{RESET}")
    print(f"{BOLD}Test cases  :{RESET} {len(cases)}")
    print(f"{BOLD}Layer       :{RESET} {layer_filter}")
    if id_filter:
        print(f"{BOLD}IDs         :{RESET} {', '.join(id_filter)}")
    print(f"{BOLD}Debug       :{RESET} {debug}")
    if log_path is not None:
        print(f"{BOLD}Log         :{RESET} {log_path}")

    error_cases = []

    for case in cases:
        text = case.get("text", "")
        if text is None:
            print(f"\n{YELLOW}[{case['id']}] Bỏ qua — text=None và chưa generate được.{RESET}")
            continue

        payload = {"text": text, "debug": debug}

        print_header(case)
        print_input(text)

        try:
            t0 = time.perf_counter()
            resp = requests.post(endpoint, json=payload, timeout=15)
            elapsed = (time.perf_counter() - t0) * 1000
        except requests.exceptions.ConnectionError:
            print(f"  {RED}✗ Không kết nối được tới {endpoint}{RESET}")
            print(f"  {DIM}Hãy chắc chắn server đang chạy: uvicorn app.main:app --port 8000{RESET}")
            sys.exit(1)
        except requests.exceptions.Timeout:
            print(f"  {RED}✗ Timeout sau 15s{RESET}")
            error_cases.append(case["id"])
            continue

        print_response(resp, elapsed)

        if resp.status_code >= 500:
            error_cases.append(case["id"])

    # Summary
    print()
    print_separator("═")
    total = len(cases)
    srv_err = len(error_cases)
    print(f"{BOLD}Tổng:{RESET} {total} test cases  |  "
          f"{RED}Server Error (5xx):{RESET} {srv_err}")
    if error_cases:
        print(f"  {RED}Các case lỗi 5xx: {', '.join(error_cases)}{RESET}")
    print_separator("═")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Xem output thực tế của API cho từng test input."
    )
    parser.add_argument(
        "--url", default="http://localhost:8000",
        help="Base URL của API (mặc định: http://localhost:8000)"
    )
    parser.add_argument(
        "--layer", default="all",
        choices=["all", "parser_api", "phone_extractor"],
        help="Lọc theo layer (mặc định: all)"
    )
    parser.add_argument(
        "--ids", nargs="*", default=[],
        metavar="ID",
        help="Chỉ chạy các ID cụ thể, vd: --ids P-01 PH-03 R-04"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Gửi debug=true trong payload để lấy thêm thông tin"
    )
    parser.add_argument(
        "--log-dir", default=str(DEFAULT_LOG_DIR),
        help="Thư mục ghi log terminal (mặc định: adds/output)"
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Suffix log file, mặc định dùng layer"
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Tắt ghi log terminal ra file"
    )
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

        run(
            url=args.url,
            layer_filter=args.layer,
            id_filter=args.ids,
            debug=args.debug,
            log_path=log_path,
        )
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    main()
