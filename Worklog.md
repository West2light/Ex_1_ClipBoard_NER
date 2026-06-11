# Worklog - 2026-06-08

## Công việc đã làm
- Bổ sung các mode benchmark cho `scripts/view_outputs.py`: `rps`, `ccu`, `both`, `feedback`.
- Ghi mode vào file JSON benchmark và thêm metadata model vào log/JSON.
- Mở rộng `/ready` để trả về `model_family`, `model_path`, `use_phobert`.
- Cập nhật NERService để hỗ trợ cả `model_quantized.onnx` và `model.onnx`.
- Tạo/cập nhật hướng dẫn V5 và các plan benchmark trong `adds/`.

## Kết quả
- PhoBERT artifact `model/phobert_ner_onnx_int8_best_hp_auth` load ready thành công.
- Benchmark đã có thể truy vết mode và cấu hình model rõ ràng hơn.
- Test gần nhất: `pytest -q` -> `76 passed, 5 skipped`.

## Hôm sau
- Viết script so sánh benchmark giữa XLM-R và PhoBERT.
- Tạo report/CSV/chart tổng quan từ 2 file benchmark JSON.
- Chạy benchmark valid-only nếu cần tách performance khỏi parser quality.
