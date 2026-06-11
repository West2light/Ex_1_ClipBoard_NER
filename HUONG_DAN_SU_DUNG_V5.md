# Huong Dan Su Dung V5 - Clipboard Parsing API dual model

## Muc tieu V5

V5 chay API parsing voi 2 model ONNX:

- XLM-R: `model/xlmr_ner_onnx_int8_best_hp_auth`
- PhoBERT: `model/phobert_ner_onnx_int8_best_hp_auth`

Pipeline van giu nguyen:

```text
raw text
-> validate + sanitize
-> phone regex + context filter
-> mask valid phones as [PHONE]
-> NER PER/ADDR/NOTE
-> post-process spans
-> resolve final fields
-> require address_raw or return address_not_found
```

## Diem can biet truoc khi chay

`app/core/config.py` hien tai resolve model theo:

```text
MODEL_FAMILY=xlmr    -> model/xlmr_ner_onnx_int8
MODEL_FAMILY=phobert -> model/phobert_ner_onnx_int8
```

Trong repo hien tai, model thuc te nam o:

```text
model/xlmr_ner_onnx_int8_best_hp_auth
model/phobert_ner_onnx_int8_best_hp_auth
```

Vi vay khi chay V5, nen set `MODEL_PATH` explicit. Dung `MODEL_FAMILY` de bat logic XLM-R/PhoBERT, va dung `MODEL_PATH` de tro toi artifact cu the.

## Kiem tra artifact

Chay:

```powershell
Get-ChildItem model\xlmr_ner_onnx_int8_best_hp_auth
Get-ChildItem model\phobert_ner_onnx_int8_best_hp_auth
```

XLM-R folder can co:

```text
model_quantized.onnx
config.json
tokenizer.json
sentencepiece.bpe.model
tokenizer_config.json
special_tokens_map.json
```

PhoBERT folder hien co:

```text
model.onnx
config.json
vocab.txt
bpe.codes
tokenizer_config.json
special_tokens_map.json
```

`NERService` V5 support 2 ten ONNX file, theo thu tu uu tien:

```text
model_quantized.onnx
model.onnx
```

Vi vay PhoBERT folder chi co `model.onnx` van chay duoc.

## Compatibility cho PhoBERT

Da chon Cach B: sua code runtime.

`app/services/ner_service.py` se load duoc ca:

```text
model_quantized.onnx
model.onnx
```

Runtime uu tien `model_quantized.onnx` neu ca 2 file cung ton tai.

## Chay XLM-R V5

PowerShell:

```powershell
$env:MODEL_FAMILY="xlmr"
$env:MODEL_PATH="model/xlmr_ner_onnx_int8_best_hp_auth"
$env:DEBUG_ENABLED="false"
$env:LOG_LEVEL="INFO"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

CMD:

```cmd
set MODEL_FAMILY=xlmr
set MODEL_PATH=model/xlmr_ner_onnx_int8_best_hp_auth
set DEBUG_ENABLED=false
set LOG_LEVEL=INFO
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Chay PhoBERT V5

PowerShell:

```powershell
$env:MODEL_FAMILY="phobert"
$env:MODEL_PATH="model/phobert_ner_onnx_int8_best_hp_auth"
$env:DEBUG_ENABLED="false"
$env:LOG_LEVEL="INFO"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

CMD:

```cmd
set MODEL_FAMILY=phobert
set MODEL_PATH=model/phobert_ner_onnx_int8_best_hp_auth
set DEBUG_ENABLED=false
set LOG_LEVEL=INFO
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

PhoBERT branch can dung `underthesea` de word-segment Vietnamese text:

```powershell
pip install underthesea>=6.8
```

Neu thieu `underthesea`, parser fallback ve raw text, nhung span PhoBERT co the kem on dinh hon.

## Chay bang file .env

Tao file `.env` o root repo.

XLM-R:

```dotenv
MODEL_FAMILY=xlmr
MODEL_PATH=model/xlmr_ner_onnx_int8_best_hp_auth
DEBUG_ENABLED=false
MAX_INPUT_LENGTH=5000
LOG_LEVEL=INFO
```

PhoBERT:

```dotenv
MODEL_FAMILY=phobert
MODEL_PATH=model/phobert_ner_onnx_int8_best_hp_auth
DEBUG_ENABLED=false
MAX_INPUT_LENGTH=5000
LOG_LEVEL=INFO
```

Sau do:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Kiem tra health va readiness

```powershell
python -c "import requests; print(requests.get('http://localhost:8000/health').status_code, requests.get('http://localhost:8000/health').text)"
python -c "import requests; print(requests.get('http://localhost:8000/ready').status_code, requests.get('http://localhost:8000/ready').text)"
```

Expected:

```json
{"status":"ready","model_family":"phobert"}
```

Hoac:

```json
{"status":"ready","model_family":"xlmr"}
```

Neu `/ready` tra ve `503 model_not_ready`, kiem tra:

- `MODEL_PATH` co dung folder khong
- folder co ONNX file runtime dang tim khong
- `config.json` co `id2label` gom `O`, `B-PER`, `I-PER`, `B-ADDR`, `I-ADDR`, `B-NOTE`, `I-NOTE`
- tokenizer files co du khong
- PhoBERT da co compatibility step `model_quantized.onnx` hoac code da support `model.onnx` chua

## Smoke test parse

PowerShell:

```powershell
@'
import requests, json

payload = {
    "text": "Nguyen Van A, 0912345678, 45 Le Loi Q1 HCM, giao buoi sang",
    "debug": False,
}
resp = requests.post("http://localhost:8000/parse", json=payload, timeout=15)
print(resp.status_code)
print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
'@ | python -
```

Expected:

- HTTP `200` neu resolve duoc address
- `phone_number` duoc extract bang rule regex
- `address_raw` khong null
- `debug` la null khi `DEBUG_ENABLED=false`

## Chay test case viewer

```powershell
python scripts/view_outputs.py --url http://localhost:8000 --layer parser_api --run-id v5_smoke
```

Output log nam o:

```text
adds/output/view_outputs_*.log
```

## Chay benchmark V5

### Feedback audit truoc

```powershell
python scripts/view_outputs.py ^
  --benchmark ^
  --benchmark-mode feedback ^
  --url http://localhost:8000 ^
  --layer parser_api ^
  --timeout 15
```

Doc:

```text
adds/feedback_benchmark.md
adds/output/view_outputs_benchmark_*.json
```

### 100 RPS va 100 CCU

```powershell
python scripts/view_outputs.py ^
  --benchmark ^
  --benchmark-mode both ^
  --url http://localhost:8000 ^
  --layer parser_api ^
  --target-rps 100 ^
  --ccu 100 ^
  --duration 60 ^
  --warmup 15 ^
  --timeout 15
```

Doc:

- `p50`
- `p95`
- `p99`
- `status_counts`
- `requests_successful`
- `requests_failed`

## Chay song song nhieu worker

```powershell
$env:MODEL_FAMILY="phobert"
$env:MODEL_PATH="model/phobert_ner_onnx_int8_best_hp_auth"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Moi worker load model rieng. Neu tang workers:

- throughput co the tang
- RAM se tang theo so worker
- p99 co the on hon neu CPU con du
- p99 co the xau hon neu CPU bi tranh chap

## So sanh XLM-R va PhoBERT

Chay lan luot:

1. XLM-R voi `MODEL_FAMILY=xlmr`, `MODEL_PATH=model/xlmr_ner_onnx_int8_best_hp_auth`
2. PhoBERT voi `MODEL_FAMILY=phobert`, `MODEL_PATH=model/phobert_ner_onnx_int8_best_hp_auth`

Voi moi model:

```powershell
python scripts/view_outputs.py --url http://localhost:8000 --layer parser_api --run-id MODEL_NAME_smoke
python scripts/view_outputs.py --benchmark --benchmark-mode feedback --url http://localhost:8000 --layer parser_api --timeout 15
python scripts/view_outputs.py --benchmark --benchmark-mode both --url http://localhost:8000 --layer parser_api --target-rps 100 --ccu 100 --duration 60 --warmup 15 --timeout 15
```

Chi ket luan performance khi:

- `/ready` da 200
- input error da duoc tach rieng
- benchmark lap lai it nhat 3 lan
- server khong doi config giua cac lan chay

## Troubleshooting nhanh

### `/ready` = 503 `model_not_ready`

Kiem tra:

```powershell
Get-ChildItem $env:MODEL_PATH
```

PhoBERT can dac biet kiem tra `model_quantized.onnx` hoac code support `model.onnx`.

### `debug_not_allowed`

Dang gui `debug=true` trong khi:

```text
DEBUG_ENABLED=false
```

Tat debug trong request hoac bat:

```powershell
$env:DEBUG_ENABLED="true"
```

### Nhieu `422 address_not_found`

Day co the la model span/resolver issue hoac workload khong co address. Chay:

```powershell
python scripts/view_outputs.py --benchmark --benchmark-mode feedback --url http://localhost:8000 --layer parser_api
```

### PhoBERT span bi lech

Kiem tra:

- `underthesea` da cai chua
- `MODEL_FAMILY=phobert` da dung chua
- input co du dau tieng Viet/word segmentation phu hop voi training khong
