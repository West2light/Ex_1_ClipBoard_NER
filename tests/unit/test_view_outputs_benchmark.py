from scripts.view_outputs import (
    load_benchmark_inputs,
    percentile,
    _categorize_feedback_record,
    _write_json_report,
    _render_feedback_markdown,
    _summarize_benchmark_results,
    _summarize_feedback_results,
)


def test_percentile_nearest_rank():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 50) == 30.0
    assert percentile(values, 99) == 50.0


def test_benchmark_inputs_skip_blank_text():
    cases = load_benchmark_inputs("parser_api", [])
    assert cases
    assert all(isinstance(case.get("text"), str) and case["text"].strip() for case in cases)


def test_benchmark_summary_counts_http_failures():
    summary = _summarize_benchmark_results(
        mode="open_loop",
        records=[
            {"case_id": "P-01", "status_code": 200, "elapsed_ms": 100.0, "error": None},
            {"case_id": "P-02", "status_code": 500, "elapsed_ms": 250.0, "error": None},
            {"case_id": "P-03", "status_code": None, "elapsed_ms": 300.0, "error": "ReadTimeout"},
        ],
        target=100.0,
        duration_sec=1.0,
        elapsed_sec=1.0,
        warmup_sec=0.0,
    )
    assert summary["requests_total"] == 3
    assert summary["requests_successful"] == 1
    assert summary["requests_failed"] == 2
    assert summary["status_counts"]["200"] == 1
    assert summary["status_counts"]["500"] == 1
    assert summary["status_counts"]["error"] == 1
    assert summary["latency_ms"]["p99"] == 250.0


def test_feedback_summary_splits_input_and_server_errors():
    summary = _summarize_feedback_results(
        records=[
            {"case_id": "P-01", "layer": "parser_api", "status_code": 200, "elapsed_ms": 10.0, "error": None, "error_code": None},
            {"case_id": "P-07", "layer": "parser_api", "status_code": 400, "elapsed_ms": 11.0, "error": None, "error_code": "invalid_input"},
            {"case_id": "R-04", "layer": "parser_api", "status_code": 422, "elapsed_ms": 12.0, "error": None, "error_code": "address_not_found"},
            {"case_id": "X-01", "layer": "parser_api", "status_code": 500, "elapsed_ms": 13.0, "error": None, "error_code": None},
            {"case_id": "X-02", "layer": "parser_api", "status_code": None, "elapsed_ms": 14.0, "error": "ReadTimeout", "error_code": None},
        ],
        duration_sec=1.0,
        elapsed_sec=1.0,
    )
    assert summary["requests_successful"] == 1
    assert summary["requests_input_error"] == 2
    assert summary["requests_server_error"] == 1
    assert summary["requests_exception"] == 1
    assert _categorize_feedback_record(summary["samples"][1]) == "input_error"
    assert _categorize_feedback_record(summary["samples"][3]) == "server_error"
    assert _categorize_feedback_record(summary["samples"][4]) == "exception"
    markdown = _render_feedback_markdown(summary)
    assert "Ưu tiên 1" in markdown
    assert "P-07" in markdown
    assert "ReadTimeout" in markdown


def test_write_json_report_uses_mode_in_filename(tmp_path):
    path = _write_json_report(
        report_dir=tmp_path,
        mode="open_loop",
        report_id="unit",
        payload={"mode": "open_loop"},
    )
    assert path.name.endswith("_open_loop_unit.json")
    assert path.exists()


def test_feedback_markdown_includes_model_metadata():
    summary = {
        "mode": "feedback",
        "requests_total": 1,
        "requests_successful": 1,
        "requests_input_error": 0,
        "requests_server_error": 0,
        "requests_exception": 0,
        "latency_ms": {"p50": 1.0, "p95": 1.0, "p99": 1.0},
        "buckets": {"input_error": [], "server_error": [], "exception": []},
        "server_model": {
            "model_family": "phobert",
            "model_path": "model/phobert_ner_onnx_int8_best_hp_auth",
            "use_phobert": True,
        },
    }
    markdown = _render_feedback_markdown(summary)
    assert "model_family" in markdown
    assert "phobert" in markdown
