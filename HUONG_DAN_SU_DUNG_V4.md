# Hướng Dẫn Sử Dụng — Clipboard Parsing API v4 (ONNX INT8 CPU)

## Tổng Quan

API phân tích văn bản đơn hàng giao hàng tiếng Việt từ clipboard, trích xuất:

| Trường | Kiểu | Bắt buộc |
|---|---|---|
| `recipient_name` | `string \| null` | Không |
| `phone_number` | `string \| null` | Không |
| `address_raw` | `string` | **Có** |
| `note` | `string \| null` | Không |

**Pipeline:**
```
văn bản thô
→ validate + sanitize
→ trích phone (regex + context filter)
→ mask phone hợp lệ → [PHONE]
→ ONNX NER (PER, ADDR, NOTE)
→ post-process spans
→ resolve fields
→ trả kết quả hoặc lỗi
```

---

## Yêu Cầu Môi Trường

| Phần mềm | Phiên bản tối thiểu |
|---|---|
| Python | 3.10+ |
| onnxruntime | 1.18+ |
| transformers | 4.40+ |
| fastapi | 0.115+ |
| uvicorn | 0.30+ |
| pydantic | 2.0+ |

> **Không cần** `torch` hay GPU. Chạy hoàn toàn trên CPU.

---

## Cài Đặt

```bash
# 1. Kích hoạt virtual environment
.\.venv\Scripts\activate          # Windows
# hoặc
source .venv/bin/activate          # Linux/macOS

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Giải nén model ONNX (chỉ cần làm 1 lần)
python -c "
import zipfile, os
for name in ['xlmr_ner_onnx_int8', 'phobert_ner_onnx_int8']:
    zp = f'model/{name}.zip'
    od = f'model/{name}'
    if os.path.exists(zp) and not os.path.exists(od):
        with zipfile.ZipFile(zp) as z:
            z.extractall(od)
        print(f'Extracted {name}')
"
```

---

## Khởi Động Server

### Mặc định (XLM-R)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Dùng PhoBERT

```bash
set MODEL_FAMILY=phobert          # Windows CMD
$env:MODEL_FAMILY="phobert"       # Windows PowerShell
export MODEL_FAMILY=phobert       # Linux/macOS

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Tuỳ Chọn Môi Trường (`.env`)

Tạo file `.env` ở thư mục gốc:

```dotenv
MODEL_FAMILY=xlmr             # xlmr | phobert
MODEL_PATH=                   # để trống để tự resolve từ MODEL_FAMILY
DEBUG_ENABLED=false           # true để bật debug output
MAX_INPUT_LENGTH=5000         # giới hạn ký tự input
LOG_LEVEL=INFO                # DEBUG | INFO | WARNING | ERROR
```

---

## Endpoints

### `GET /health` — Liveness Probe

Kiểm tra process đang chạy. Không kiểm tra model.

```bash
curl http://localhost:8000/health
```

**Response 200:**
```json
{"status": "ok"}
```

---

### `GET /ready` — Readiness Probe

Kiểm tra model đã load xong và sẵn sàng nhận request.

```bash
curl http://localhost:8000/ready
```

**Response 200 (ready):**
```json
{"status": "ready", "model_family": "xlmr"}
```

**Response 503 (chưa sẵn sàng):**
```json
{
  "error": {
    "code": "model_not_ready",
    "message": "The parsing model is not ready."
  },
  "debug": null
}
```

---

### `POST /parse` — Phân Tích Văn Bản

#### Request Body

```json
{
  "text": "Nguyễn Văn A, 0912345678\n45 Lê Lợi Q1 HCM\ngiao trước 10h sáng",
  "debug": false
}
```

| Tham số | Kiểu | Mô tả |
|---|---|---|
| `text` | `string` | Văn bản cần phân tích (bắt buộc) |
| `debug` | `bool` | Trả về thông tin debug (mặc định `false`) |

#### Ví Dụ — cURL

```bash
curl -X POST http://localhost:8000/parse \
  -H "Content-Type: application/json" \
  -d '{"text": "Nguyễn Văn A, 0912345678\n45 Lê Lợi Q1 HCM\ngiao trước 10h sáng"}'
```

#### Response 200 — Thành Công

```json
{
  "recipient_name": "Nguyễn Văn A",
  "phone_number": "0912345678",
  "address_raw": "45 Lê Lợi Q1 HCM",
  "note": "giao trước 10h sáng",
  "debug": null
}
```

#### Response 200 — Phone Không Hợp Lệ Nhưng Có Địa Chỉ

```json
{
  "recipient_name": "Trần B",
  "phone_number": null,
  "address_raw": "22 Hai Bà Trưng Q1",
  "note": null,
  "debug": null
}
```

> Phone không hợp lệ **không làm fail** request khi đã có địa chỉ.

---

## Bảng Lỗi

| HTTP | Code | Nguyên Nhân |
|---|---|---|
| 400 | `invalid_input` | Body sai, thiếu `text`, `text` rỗng, có field lạ |
| 400 | `input_too_long` | `text` vượt quá 5000 ký tự Unicode |
| 403 | `debug_not_allowed` | `debug=true` khi server chưa bật `DEBUG_ENABLED` |
| 422 | `address_not_found` | Không tìm được địa chỉ hợp lệ |
| 503 | `model_not_ready` | Model chưa load xong hoặc lỗi khởi tạo |
| 500 | `internal_error` | Lỗi nội bộ không xác định |

**Envelope lỗi:**
```json
{
  "error": {
    "code": "address_not_found",
    "message": "A delivery address could not be resolved."
  },
  "debug": null
}
```

---

## Ví Dụ Thực Tế

### Python

```python
import httpx

resp = httpx.post(
    "http://localhost:8000/parse",
    json={"text": "Chị Lan, SĐT: 0901234567\nGiao tới: 123 Điện Biên Phủ P15 Q.Bình Thạnh"}
)
data = resp.json()
print(data["recipient_name"])  # Chị Lan
print(data["phone_number"])    # 0901234567
print(data["address_raw"])     # 123 Điện Biên Phủ P15 Q.Bình Thạnh
```

### JavaScript (fetch)

```javascript
const res = await fetch("http://localhost:8000/parse", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    text: "Anh Minh, 0912345678\n88 Nguyễn Du Q1 HCM"
  })
});
const data = await res.json();
console.log(data.recipient_name); // Anh Minh
console.log(data.address_raw);    // 88 Nguyễn Du Q1 HCM
```

---

## Chạy Tests

```bash
# Toàn bộ test suite (không cần server)
python -m pytest tests/ -v

# Chỉ unit tests
python -m pytest tests/unit/ -v

# Chỉ API contract tests
python -m pytest tests/api/ -v

# Kết quả mong đợi: 51 passed
```

---

## Cấu Trúc Thư Mục

```
app/
  main.py                  # FastAPI app factory + lifespan
  schemas.py               # Pydantic request/response models
  api/
    routes.py              # POST /parse, GET /health, GET /ready
  core/
    config.py              # Settings (env-driven)
    errors.py              # Typed error classes + handlers
    logging.py             # Structured logging setup
  services/
    sanitizer.py           # Làm sạch văn bản + offset map
    phone_extractor.py     # Trích xuất và lọc số điện thoại
    ner_service.py         # ONNX NER inference (XLM-R / PhoBERT)
    phobert_adapter.py     # Tách từ tiếng Việt cho PhoBERT
    postprocessor.py       # Lắp ráp BIO spans → candidates
    resolver.py            # Chọn field cuối cùng từ candidates
    parser_service.py      # Orchestration toàn bộ pipeline

model/
  xlmr_ner_onnx_int8/      # XLM-R ONNX INT8 (~218MB)
  phobert_ner_onnx_int8/   # PhoBERT ONNX INT8 (~212MB)

tests/
  unit/                    # Unit tests (không cần server)
  api/                     # API contract tests (dùng httpx + mock)
```

---

## Hiệu Năng (Colab CPU)

| Model | Runtime | ms/call |
|---|---|---|
| XLM-R FP32 | PyTorch CPU | ~114ms |
| XLM-R INT8 | ONNX Runtime | ~30ms |
| PhoBERT INT8 | ONNX Runtime | ~30ms |
| **Speedup** | | **~3.8×** |

SLA: p95 ≤ 2000ms (R-06).

---

## Lưu Ý Quan Trọng

> [!WARNING]
> Không lưu trữ văn bản khách hàng, số điện thoại, hay tên người dùng vào log hoặc file.  
> Debug output chỉ chứa offsets, flags và latency — không chứa nội dung text thực.

> [!NOTE]
> Khi dùng `MODEL_FAMILY=phobert`, cần cài thêm `underthesea`:
> ```bash
> pip install underthesea>=6.8
> ```
> Nếu không có `underthesea`, hệ thống tự fallback về text gốc (không tách từ).

> [!TIP]
> Để chạy nhiều worker song song:
> ```bash
> uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
> ```
> Mỗi worker tải model riêng (~210MB RAM mỗi worker).
