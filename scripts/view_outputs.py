"""
view_outputs.py — Gửi từng test case lên API và in output thực tế ra màn hình.
Mục đích: quan sát response thô, KHÔNG so sánh với expected.

Chạy:
    python scripts/view_outputs.py [--url http://localhost:8000] [--ids P-01 PH-03]
                                   [--layer parser_api|phone_extractor|all]
                                   [--debug]
"""

import argparse
import asyncio
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
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


def load_benchmark_inputs(layer_filter: str, id_filter: list[str]) -> list[dict]:
    """Return only runnable cases for benchmark mode."""
    cases = load_inputs(layer_filter, id_filter)
    runnable = []
    for case in cases:
        text = case.get("text")
        if isinstance(text, str) and text.strip():
            runnable.append(case)
    return runnable


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


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if pct <= 0:
        return ordered[0]
    if pct >= 100:
        return ordered[-1]
    rank = math.ceil((pct / 100.0) * len(ordered)) - 1
    rank = max(0, min(rank, len(ordered) - 1))
    return ordered[rank]


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _format_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def _make_report_path(log_dir: Path, mode: str, run_id: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{mode}_{run_id}" if run_id else mode
    base = log_dir / f"view_outputs_benchmark_{stamp}_{suffix}.json"
    if not base.exists():
        return base

    idx = 2
    while True:
        candidate = log_dir / f"view_outputs_benchmark_{stamp}_{suffix}_{idx}.json"
        if not candidate.exists():
            return candidate
        idx += 1


def _write_json_report(
    report_dir: Path,
    mode: str,
    report_id: str | None,
    payload: dict[str, Any],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = _make_report_path(report_dir, mode, report_id)
    with report_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return report_path


def _benchmark_payload(case: dict, debug: bool) -> dict[str, Any]:
    return {"text": case.get("text", ""), "debug": debug}


async def _fetch_ready_metadata(
    client: httpx.AsyncClient,
    base_url: str,
) -> dict[str, Any]:
    resp = await client.get(base_url.rstrip("/") + "/ready")
    if resp.status_code != 200:
        raise RuntimeError(f"Server not ready: HTTP {resp.status_code} {resp.text[:200]}")

    body = resp.json()
    return {
        "status": body.get("status"),
        "model_family": body.get("model_family"),
        "model_path": body.get("model_path"),
        "use_phobert": body.get("use_phobert"),
    }


async def _send_benchmark_request(
    client: httpx.AsyncClient,
    endpoint: str,
    case: dict,
    debug: bool,
    scheduled_at: float | None = None,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    if scheduled_at is not None:
        delay = scheduled_at - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)

    started = loop.time()
    payload = _benchmark_payload(case, debug)
    status_code = None
    error = None
    error_code = None
    try:
        resp = await client.post(endpoint, json=payload)
        status_code = resp.status_code
        if status_code >= 400:
            try:
                body = resp.json()
                err = body.get("error")
                if isinstance(err, dict):
                    error_code = err.get("code")
            except Exception:
                pass
    except Exception as exc:
        error = exc.__class__.__name__

    elapsed_ms = (loop.time() - started) * 1000
    return {
        "case_id": case["id"],
        "layer": case.get("layer"),
        "status_code": status_code,
        "elapsed_ms": elapsed_ms,
        "error": error,
        "error_code": error_code,
    }


async def _run_open_loop(
    client: httpx.AsyncClient,
    endpoint: str,
    cases: list[dict],
    target_rps: float,
    duration_sec: float,
    debug: bool,
) -> list[dict[str, Any]]:
    if target_rps <= 0:
        raise ValueError("target_rps must be > 0")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be > 0")

    loop = asyncio.get_running_loop()
    total_requests = max(1, math.ceil(target_rps * duration_sec))
    interval = 1.0 / target_rps
    start = loop.time() + 0.1

    tasks = []
    for idx in range(total_requests):
        case = cases[idx % len(cases)]
        scheduled_at = start + idx * interval
        tasks.append(
            asyncio.create_task(
                _send_benchmark_request(
                    client=client,
                    endpoint=endpoint,
                    case=case,
                    debug=debug,
                    scheduled_at=scheduled_at,
                )
            )
        )
    return await asyncio.gather(*tasks)


async def _run_closed_loop(
    client: httpx.AsyncClient,
    endpoint: str,
    cases: list[dict],
    ccu: int,
    duration_sec: float,
    debug: bool,
) -> list[dict[str, Any]]:
    if ccu <= 0:
        raise ValueError("ccu must be > 0")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be > 0")

    loop = asyncio.get_running_loop()
    deadline = loop.time() + duration_sec
    results: list[dict[str, Any]] = []
    results_lock = asyncio.Lock()

    async def worker(worker_idx: int):
        iteration = 0
        while loop.time() < deadline:
            case = cases[(worker_idx + iteration) % len(cases)]
            record = await _send_benchmark_request(
                client=client,
                endpoint=endpoint,
                case=case,
                debug=debug,
            )
            async with results_lock:
                results.append(record)
            iteration += 1

    await asyncio.gather(*(worker(i) for i in range(ccu)))
    return results


def _summarize_benchmark_results(
    mode: str,
    records: list[dict[str, Any]],
    target: float,
    duration_sec: float,
    elapsed_sec: float,
    warmup_sec: float,
    ccu: int | None = None,
) -> dict[str, Any]:
    completed = [r for r in records if r["error"] is None and r["status_code"] is not None]
    successful = [r for r in completed if 200 <= int(r["status_code"]) < 300]
    failed = [r for r in records if r["error"] is not None or r["status_code"] is None or int(r["status_code"]) >= 300]
    latencies = [float(r["elapsed_ms"]) for r in completed]
    achieved_rps = len(records) / elapsed_sec if elapsed_sec > 0 else None

    status_counts: dict[str, int] = {}
    for record in records:
        key = "error" if record["status_code"] is None else str(record["status_code"])
        status_counts[key] = status_counts.get(key, 0) + 1

    return {
        "mode": mode,
        "target": target,
        "ccu": ccu,
        "duration_sec": duration_sec,
        "elapsed_sec": elapsed_sec,
        "warmup_sec": warmup_sec,
        "requests_total": len(records),
        "requests_completed": len(completed),
        "requests_successful": len(successful),
        "requests_failed": len(failed),
        "achieved_rps": achieved_rps,
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "mean": _safe_mean(latencies),
            "p50": percentile(latencies, 50),
            "p90": percentile(latencies, 90),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "max": max(latencies) if latencies else None,
        },
        "status_counts": status_counts,
        "samples": records,
    }


def _categorize_feedback_record(record: dict[str, Any]) -> str:
    status_code = record.get("status_code")
    if record.get("error") is not None or status_code is None:
        return "exception"
    if 400 <= int(status_code) < 500:
        return "input_error"
    if int(status_code) >= 500:
        return "server_error"
    return "success"


def _summarize_feedback_results(
    records: list[dict[str, Any]],
    duration_sec: float,
    elapsed_sec: float,
    warmup_sec: float = 0.0,
) -> dict[str, Any]:
    categories = {
        "success": [],
        "input_error": [],
        "server_error": [],
        "exception": [],
    }
    for record in records:
        category = _categorize_feedback_record(record)
        categories[category].append(record)

    latencies = [float(r["elapsed_ms"]) for r in records if r.get("status_code") is not None and r.get("error") is None]
    status_counts: dict[str, int] = {}
    for record in records:
        key = "error" if record.get("status_code") is None else str(record["status_code"])
        status_counts[key] = status_counts.get(key, 0) + 1

    return {
        "mode": "feedback",
        "duration_sec": duration_sec,
        "elapsed_sec": elapsed_sec,
        "warmup_sec": warmup_sec,
        "requests_total": len(records),
        "requests_successful": len(categories["success"]),
        "requests_input_error": len(categories["input_error"]),
        "requests_server_error": len(categories["server_error"]),
        "requests_exception": len(categories["exception"]),
        "status_counts": status_counts,
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "mean": _safe_mean(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "max": max(latencies) if latencies else None,
        },
        "buckets": categories,
        "samples": records,
    }


def _render_feedback_markdown(summary: dict[str, Any]) -> str:
    latency = summary["latency_ms"]
    model_info = summary.get("server_model") or {}
    lines = [
        "# Benchmark feedback audit",
        "",
        "## Summary",
        f"- model_family: {model_info.get('model_family')}",
        f"- model_path: {model_info.get('model_path')}",
        f"- use_phobert: {model_info.get('use_phobert')}",
        f"- mode: {summary['mode']}",
        f"- total requests: {summary['requests_total']}",
        f"- success: {summary['requests_successful']}",
        f"- input errors: {summary['requests_input_error']}",
        f"- server errors: {summary['requests_server_error']}",
        f"- exceptions: {summary['requests_exception']}",
        f"- p50: {_format_ms(latency['p50'])} ms",
        f"- p95: {_format_ms(latency['p95'])} ms",
        f"- p99: {_format_ms(latency['p99'])} ms",
        "",
        "## Ưu tiên 1 - Tách lỗi input khỏi lỗi server",
        "",
        "### Input errors",
    ]

    input_rows = summary["buckets"]["input_error"]
    if input_rows:
        lines.extend([
            "| case_id | layer | status | error_code | latency_ms |",
            "|---|---|---:|---|---:|",
        ])
        for record in sorted(input_rows, key=lambda r: (int(r.get("status_code") or 0), r["case_id"])):
            lines.append(
                f"| {record['case_id']} | {record.get('layer','')} | {record.get('status_code')} | "
                f"{record.get('error_code') or ''} | {float(record.get('elapsed_ms', 0.0)):.1f} |"
            )
    else:
        lines.append("_None_")

    lines.extend(["", "### Server/runtime errors"])
    server_rows = summary["buckets"]["server_error"] + summary["buckets"]["exception"]
    if server_rows:
        lines.extend([
            "| case_id | layer | status | error | latency_ms |",
            "|---|---|---:|---|---:|",
        ])
        for record in sorted(server_rows, key=lambda r: (r.get("status_code") is None, int(r.get("status_code") or 0), r["case_id"])):
            lines.append(
                f"| {record['case_id']} | {record.get('layer','')} | {record.get('status_code') or ''} | "
                f"{record.get('error') or record.get('error_code') or ''} | {float(record.get('elapsed_ms', 0.0)):.1f} |"
            )
    else:
        lines.append("_None_")

    lines.extend(["", "### Success cases", f"- count: {summary['requests_successful']}"])
    return "\n".join(lines) + "\n"


def _append_feedback_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = _render_feedback_markdown(summary)
    has_existing = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="\n") as f:
        if has_existing:
            f.write(f"\n\n---\n\n## Auto report {stamp}\n\n")
        else:
            f.write(f"## Auto report {stamp}\n\n")
        f.write(content)


def print_model_context(model_info: dict[str, Any]) -> None:
    print()
    print_separator("═")
    print(f"{BOLD}Model    :{RESET} {model_info.get('model_family')}")
    print(f"{BOLD}Path     :{RESET} {model_info.get('model_path')}")
    print(f"{BOLD}PhoBERT  :{RESET} {model_info.get('use_phobert')}")
    print_separator("═")


def print_benchmark_summary(summary: dict[str, Any]) -> None:
    latency = summary["latency_ms"]
    print()
    print_separator("═")
    print(f"{BOLD}Benchmark:{RESET} {summary['mode']}")
    print(f"{BOLD}Target   :{RESET} {summary['target']}")
    if summary.get("ccu") is not None:
        print(f"{BOLD}CCU      :{RESET} {summary['ccu']}")
    print(f"{BOLD}Duration :{RESET} {summary['duration_sec']:.1f}s")
    print(f"{BOLD}Elapsed  :{RESET} {summary['elapsed_sec']:.1f}s")
    print(f"{BOLD}Warmup   :{RESET} {summary['warmup_sec']:.1f}s")
    print(f"{BOLD}Requests :{RESET} {summary['requests_total']}")
    print(f"{BOLD}Success  :{RESET} {summary['requests_successful']}  "
          f"{BOLD}Fail:{RESET} {summary['requests_failed']}")
    print(f"{BOLD}Achieved :{RESET} {summary['achieved_rps']:.1f} rps")
    print(f"{BOLD}Latency  :{RESET} p50={_format_ms(latency['p50'])} ms  "
          f"p95={_format_ms(latency['p95'])} ms  "
          f"p99={_format_ms(latency['p99'])} ms")
    print(f"{BOLD}Status   :{RESET} {summary['status_counts']}")
    print_separator("═")


def print_feedback_summary(summary: dict[str, Any]) -> None:
    latency = summary["latency_ms"]
    print()
    print_separator("═")
    print(f"{BOLD}Benchmark:{RESET} feedback")
    print(f"{BOLD}Requests :{RESET} {summary['requests_total']}")
    print(f"{BOLD}Success  :{RESET} {summary['requests_successful']}")
    print(f"{BOLD}InputErr :{RESET} {summary['requests_input_error']}")
    print(f"{BOLD}ServerErr:{RESET} {summary['requests_server_error']}")
    print(f"{BOLD}Except   :{RESET} {summary['requests_exception']}")
    print(f"{BOLD}Latency  :{RESET} p50={_format_ms(latency['p50'])} ms  "
          f"p95={_format_ms(latency['p95'])} ms  "
          f"p99={_format_ms(latency['p99'])} ms")
    print(f"{BOLD}Status   :{RESET} {summary['status_counts']}")
    print_separator("═")


async def run_benchmark(
    url: str,
    layer_filter: str,
    id_filter: list[str],
    debug: bool,
    benchmark_mode: str,
    target_rps: float,
    ccu: int,
    duration_sec: float,
    warmup_sec: float,
    timeout_sec: float,
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    cases = load_benchmark_inputs(layer_filter if layer_filter != "all" else "parser_api", id_filter)
    if not cases:
        print(f"{RED}Khong tim thay benchmark case nao phu hop.{RESET}")
        sys.exit(1)

    endpoint = url.rstrip("/") + "/parse"
    print(f"\n{BOLD}API endpoint:{RESET} {CYAN}{endpoint}{RESET}")
    print(f"{BOLD}Benchmark  :{RESET} {benchmark_mode}")
    print(f"{BOLD}Cases      :{RESET} {len(cases)}")
    print(f"{BOLD}Target RPS :{RESET} {target_rps}")
    print(f"{BOLD}CCU        :{RESET} {ccu}")
    print(f"{BOLD}Duration   :{RESET} {duration_sec:.1f}s")
    print(f"{BOLD}Warmup     :{RESET} {warmup_sec:.1f}s")
    print(f"{BOLD}Timeout    :{RESET} {timeout_sec:.1f}s")
    print(f"{BOLD}Debug      :{RESET} {debug}")
    if log_path is not None:
        print(f"{BOLD}Log        :{RESET} {log_path}")

    all_summaries: list[dict[str, Any]] = []
    timeout = httpx.Timeout(timeout_sec)
    limits = httpx.Limits(max_connections=max(100, ccu * 2), max_keepalive_connections=max(20, ccu))

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        model_info = await _fetch_ready_metadata(client, url)
        print_model_context(model_info)

        async def warmup_open_loop():
            if warmup_sec <= 0:
                return
            await _run_open_loop(client, endpoint, cases, target_rps, warmup_sec, debug)

        async def warmup_closed_loop():
            if warmup_sec <= 0:
                return
            await _run_closed_loop(client, endpoint, cases, ccu, warmup_sec, debug)

        if benchmark_mode in {"rps", "both"}:
            await warmup_open_loop()
            phase_start = time.perf_counter()
            records = await _run_open_loop(client, endpoint, cases, target_rps, duration_sec, debug)
            phase_elapsed = time.perf_counter() - phase_start
            summary = _summarize_benchmark_results(
                mode="open_loop",
                records=records,
                target=target_rps,
                duration_sec=duration_sec,
                elapsed_sec=phase_elapsed,
                warmup_sec=warmup_sec,
            )
            summary["server_model"] = model_info
            print_benchmark_summary(summary)
            all_summaries.append(summary)

        if benchmark_mode in {"ccu", "both"}:
            await warmup_closed_loop()
            phase_start = time.perf_counter()
            records = await _run_closed_loop(client, endpoint, cases, ccu, duration_sec, debug)
            phase_elapsed = time.perf_counter() - phase_start
            summary = _summarize_benchmark_results(
                mode="closed_loop",
                records=records,
                target=float(ccu),
                duration_sec=duration_sec,
                elapsed_sec=phase_elapsed,
                warmup_sec=warmup_sec,
                ccu=ccu,
            )
            summary["server_model"] = model_info
            print_benchmark_summary(summary)
            all_summaries.append(summary)

    return all_summaries


async def run_feedback_benchmark(
    url: str,
    layer_filter: str,
    id_filter: list[str],
    debug: bool,
    timeout_sec: float,
    feedback_path: Path,
    log_path: Path | None = None,
) -> dict[str, Any]:
    cases = load_inputs(layer_filter if layer_filter != "all" else "parser_api", id_filter)
    if not cases:
        print(f"{RED}Khong tim thay benchmark case nao phu hop.{RESET}")
        sys.exit(1)

    endpoint = url.rstrip("/") + "/parse"
    print(f"\n{BOLD}API endpoint:{RESET} {CYAN}{endpoint}{RESET}")
    print(f"{BOLD}Benchmark  :{RESET} feedback")
    print(f"{BOLD}Cases      :{RESET} {len(cases)}")
    print(f"{BOLD}Timeout    :{RESET} {timeout_sec:.1f}s")
    print(f"{BOLD}Debug      :{RESET} {debug}")
    if log_path is not None:
        print(f"{BOLD}Log        :{RESET} {log_path}")
    print(f"{BOLD}Report     :{RESET} {feedback_path}")

    timeout = httpx.Timeout(timeout_sec)
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        model_info = await _fetch_ready_metadata(client, url)
        print_model_context(model_info)

        for case in cases:
            record = await _send_benchmark_request(client, endpoint, case, debug)
            records.append(record)
    elapsed_sec = time.perf_counter() - started

    summary = _summarize_feedback_results(
        records=records,
        duration_sec=elapsed_sec,
        elapsed_sec=elapsed_sec,
        warmup_sec=0.0,
    )
    summary["server_model"] = model_info
    print_feedback_summary(summary)
    _append_feedback_report(feedback_path, summary)
    return summary


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
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Chạy benchmark P99 thay vì in từng response"
    )
    parser.add_argument(
        "--benchmark-mode", default="both",
        choices=["rps", "ccu", "both", "feedback"],
        help="Kiểu benchmark: 100 RPS, 100 CCU, cả hai, hoặc feedback audit"
    )
    parser.add_argument(
        "--target-rps", type=float, default=100.0,
        help="Mục tiêu RPS cho open-loop benchmark"
    )
    parser.add_argument(
        "--ccu", type=int, default=100,
        help="Số concurrent users cho closed-loop benchmark"
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Thời lượng đo chính, đơn vị giây"
    )
    parser.add_argument(
        "--warmup", type=float, default=15.0,
        help="Thời lượng warmup trước khi đo, đơn vị giây"
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0,
        help="HTTP timeout cho mỗi request, đơn vị giây"
    )
    parser.add_argument(
        "--report-dir", default=str(DEFAULT_LOG_DIR),
        help="Thư mục ghi JSON report benchmark"
    )
    parser.add_argument(
        "--report-id", default=None,
        help="Suffix cho file report benchmark"
    )
    parser.add_argument(
        "--feedback-path", default=str(REPO_ROOT / "adds" / "feedback_benchmark.md"),
        help="Thu muc ghi markdown feedback benchmark"
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

        if args.benchmark:
            if args.benchmark_mode == "feedback":
                feedback_path = Path(args.feedback_path)
                if not feedback_path.is_absolute():
                    feedback_path = REPO_ROOT / feedback_path
                summary = asyncio.run(
                    run_feedback_benchmark(
                        url=args.url,
                        layer_filter=args.layer,
                        id_filter=args.ids,
                        debug=args.debug,
                        timeout_sec=args.timeout,
                        feedback_path=feedback_path,
                        log_path=log_path,
                    )
                )
                summaries = [summary]
            else:
                summaries = asyncio.run(
                    run_benchmark(
                        url=args.url,
                        layer_filter=args.layer,
                        id_filter=args.ids,
                        debug=args.debug,
                        benchmark_mode=args.benchmark_mode,
                        target_rps=args.target_rps,
                        ccu=args.ccu,
                        duration_sec=args.duration,
                        warmup_sec=args.warmup,
                        timeout_sec=args.timeout,
                        log_path=log_path,
                    )
                )
            report_dir = Path(args.report_dir)
            if not report_dir.is_absolute():
                report_dir = REPO_ROOT / report_dir

            common_payload = {
                "url": args.url,
                "layer": args.layer,
                "ids": args.ids,
                "benchmark_mode": args.benchmark_mode,
                "target_rps": args.target_rps,
                "ccu": args.ccu,
                "duration_sec": args.duration,
                "warmup_sec": args.warmup,
                "timeout_sec": args.timeout,
                "server_model": summaries[0].get("server_model") if summaries else None,
            }

            report_paths: list[Path] = []
            combined_payload = {**common_payload, "summaries": summaries}
            combined_path = _write_json_report(
                report_dir=report_dir,
                mode=args.benchmark_mode,
                report_id=args.report_id,
                payload=combined_payload,
            )
            report_paths.append(combined_path)

            if len(summaries) > 1:
                for summary in summaries:
                    per_mode_payload = {
                        **common_payload,
                        "report_mode": summary["mode"],
                        "summaries": [summary],
                    }
                    per_mode_path = _write_json_report(
                        report_dir=report_dir,
                        mode=summary["mode"],
                        report_id=args.report_id,
                        payload=per_mode_payload,
                    )
                    report_paths.append(per_mode_path)

            print(f"{BOLD}Report   :{RESET} {', '.join(str(path) for path in report_paths)}")
        else:
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
