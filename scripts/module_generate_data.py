"""Generate synthetic NER training data via Claude Haiku 4.5 few-shot.

Pipeline:
  1. Load seed examples from data/seed/seed.jsonl
  2. Few-shot Claude với system prompt + seed (cached)
  3. Mỗi batch yêu cầu N example theo 1 variation pattern
  4. Validate: phone extract + mask → substring search → char offset
  5. Reject example với offset không khớp, log reject rate
  6. Append vào data/synthetic/synth_v1.jsonl

Usage:
  uv run python -m training.generate_data --total 200 --batch-size 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import anthropic
import httpx

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - only hit when dependency is missing
    load_dotenv = None

DEFAULT_MODEL_ID = "claude-haiku-4-5-20251001"
DEFAULT_OPENROUTER_MODEL_ID = "z-ai/glm-4.5-air:free"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

# Phone regex từ legacy_rule_parser, sẽ shared khi build phone_extractor.py
_SEP = r"[\s\-\.\(\)]*"
_PHONE_RE = re.compile(
    rf"(?<!\d)(\+?84|0){_SEP}([3-9](?:{_SEP}\d){{8}})(?!\d)"
)


def mask_phone(text: str) -> str:
    """Replace VN mobile numbers với [PHONE] placeholder."""
    masked = text
    for m in reversed(list(_PHONE_RE.finditer(text))):
        digits = re.sub(r"\D", "", m.group())
        if digits.startswith("84") and len(digits) == 11:
            digits = "0" + digits[2:]
        if not (len(digits) == 10 and digits[0] == "0" and digits[1] in "3456789"):
            continue
        masked = masked[: m.start()] + "[PHONE]" + masked[m.end():]
    return masked


SYSTEM_PROMPT = """Bạn là data generator cho tiếng Việt clipboard parsing. Task: sinh đoạn text mô phỏng clipboard mà người dùng paste vào form tạo đơn delivery (từ chat, ghi chú, paste từ Facebook/Zalo, …), kèm nhãn NER chính xác.

# Entity schema

3 loại entity:
- **PER** — họ tên người NHẬN hàng (recipient). KHÔNG label tên người gửi (sender).
- **ADDR** — địa chỉ giao hàng. Có thể là số nhà + đường + phường + quận + tỉnh, hoặc landmark/POI (chung cư, KĐT, trường, công ty).
- **NOTE** — ghi chú về cách giao (thời gian, fragile, gọi trước, ngõ ngách khó tìm, ...).

# Phân biệt sender vs recipient (E-06 rule)

Khi text dạng hội thoại, suy ra ai là recipient từ context:
- "chị Mai ơi giao cho mình nha, mình ở 22 Ngô Quyền" → Mai là sender (người nhắn cho shop), "mình" là recipient → KHÔNG label PER (không có tên cụ thể của recipient)
- "ship cho Hùng" / "giao cho Lan" / "tên: X" / "tên người nhận:" → đây mới là PER
- Nếu không chắc, không label PER (recall trade-off precision).

# Output format

Output JSON theo schema đã cho. Mỗi example:
- `text`: clipboard text (CÓ thể chứa SĐT thực, model sẽ tự mask sau)
- `entities`: list của `{substring, label}` — substring PHẢI là exact substring (case-sensitive) trong text. Không cần offset, code sẽ tính.

# Diversity yêu cầu

Tạo đa dạng:
- **Format**: có label rõ (Tên:/ĐT:/Địa chỉ:/Ghi chú:), separator-only (comma/dash/pipe), conversational ("ship cho", "giao tới"), multi-line (newline-separated), messy (no punctuation)
- **Naming style**: "Nguyễn Văn A" (đầy đủ), "Hùng" (1 từ), "anh Bình"/"chị Lan" (prefix), không dấu ("Nguyen Van A")
- **Address style**: số + đường + quận đầy đủ, abbrev (Q1/Q.1/P5/TP.HCM), landmark (Vincom Center, Royal City, KĐT Vinhomes), đường ngõ ngách HN
- **Note style**: time ("trước 10h", "sau 5h chiều"), fragile ("hàng dễ vỡ"), call-before ("gọi trước 30p"), special ("cồng kềnh", "thanh toán shipper")
- **Missing field**: đôi khi không có PER, hoặc không có NOTE, hoặc cả 2

# CRITICAL — KHÔNG label

- **KHÔNG label SĐT** — phone extract bằng regex riêng
- **KHÔNG label PER cho sender** (xem E-06 rule trên)
- Số nhà không phải phone (vd "nhà số 123") → KHÔNG label PER/PHONE
- "mã đơn: 0912..." → context "mã đơn" → KHÔNG coi là PHONE (nhưng đằng nào cũng không cần label PHONE)
"""


def _seed_block(seed_path: Path, n: int = 12) -> str:
    """Format seed examples thành text block để Claude tham khảo."""
    # Use utf-8-sig so files with or without BOM decode correctly on Windows.
    with open(seed_path, encoding="utf-8-sig") as f:
        seeds = []
        for line in f:
            if not line.strip():
                continue
            seeds.append(json.loads(line))
            if len(seeds) >= n:
                break
    parts = ["# Reference examples (đã được con người label cẩn thận)\n"]
    for i, ex in enumerate(seeds, 1):
        text = ex["text"]
        ents = [
            {"substring": text[e["start"]:e["end"]], "label": e["label"]}
            for e in ex["entities"]
        ]
        parts.append(f"Example {i}:")
        parts.append(json.dumps({"text": text, "entities": ents}, ensure_ascii=False))
        parts.append("")
    return "\n".join(parts)


VARIATIONS = {
    "labels_explicit": "Format với label rõ ràng: 'Tên:', 'SĐT:'/'ĐT:', 'Địa chỉ:', 'Ghi chú:'. Separator là pipe '|' hoặc newline.",
    "separator_only": "Format chỉ dùng separator (comma, dash, slash), KHÔNG có label. Như list dữ liệu.",
    "conversational": "Style hội thoại tự nhiên: 'ship cho X', 'giao cho Y', 'mình order', 'cho mình gửi'. Có thể có nhiều fluff.",
    "multiline_simple": "Mỗi field 1 dòng (newline-separated), 3-4 dòng. KHÔNG có label.",
    "messy_lowercase": "Text messy: lowercase toàn bộ, không dấu hoặc dấu sai, viết tắt cực ngắn (q1, p5, tphcm, kdc, kp).",
    "quoted_note": "Note nằm trong dấu ngoặc kép, vd: 'nhớ ghi \"hàng dễ vỡ\" lên kiện', 'kiện ghi \"không gấp\"'.",
    "address_only": "CHỈ có address (không có tên, không có phone, có thể có note). Style: 'giao tới X' / 'shop ở X' / chỉ 1 chuỗi địa chỉ.",
    "name_phone_only": "CHỈ có tên + phone (KHÔNG có địa chỉ trong text). Đây là edge case sẽ trigger error address_not_found.",
    "with_product_info": "Có thông tin sản phẩm/đơn hàng mixed in (vd: '5 áo size M màu đen, ship cho X...'). Model phải bỏ qua thông tin sản phẩm.",
    "landmark_address": "Address chính là landmark / POI: tên chung cư (Royal City, Sunrise), KĐT, trường học, công ty. Có thể có chi tiết tầng / căn hộ.",
    "sender_vs_recipient": "Conversational hội thoại có chứa tên sender (vd 'chị Mai ơi', 'em ơi') — KHÔNG label PER cho sender. Recipient có thể là 'mình' (không có tên cụ thể) hoặc tên rõ ràng đứng sau 'cho'/'gửi cho'.",
    "multi_phone": "Có 2 SĐT (vd 'Lan 0901..., shop 0987...' hoặc 'liên hệ A: 0901..., B: 0987...'). Trong text, recipient là người có name gần phone đầu tiên.",
}


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "substring": {"type": "string"},
                                "label": {"type": "string", "enum": ["PER", "ADDR", "NOTE"]},
                            },
                            "required": ["substring", "label"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["text", "entities"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["examples"],
    "additionalProperties": False,
}


def _extract_json_text(response_text: str) -> str:
    """Return a JSON object string, tolerating markdown fences from non-strict models."""
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    if text.startswith("{"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    return text


def _load_json_object(response_text: str) -> dict:
    text = _extract_json_text(response_text)
    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(text)
    if not isinstance(data, dict):
        raise json.JSONDecodeError("top-level JSON value is not an object", text, 0)
    return data


def parse_and_validate(response_text: str) -> tuple[list[dict], list[str]]:
    """Parse Claude response → validated examples (with char-offset entities).

    Returns: (valid_examples, reject_reasons)
    """
    data = _load_json_object(response_text)
    valid: list[dict] = []
    reasons: list[str] = []

    for ex in data.get("examples", []):
        text = mask_phone(ex["text"])
        ents = []
        ok = True
        cursor = 0
        # Sort entities theo thứ tự xuất hiện trong text raw để cursor advance đúng
        # Claude thường đã order theo physical position, nhưng để chắc, ta search độc lập
        seen_spans = []
        for ent in ex["entities"]:
            sub = ent["substring"]
            label = ent["label"]
            # Search lần đầu, sau đó sort
            pos = text.find(sub)
            if pos == -1:
                ok = False
                reasons.append(f"substring {sub!r} not in masked text")
                break
            seen_spans.append({"start": pos, "end": pos + len(sub), "label": label, "sub": sub})

        if not ok:
            continue

        # Sort by start position
        seen_spans.sort(key=lambda s: s["start"])

        # Check no overlap
        for i in range(1, len(seen_spans)):
            if seen_spans[i]["start"] < seen_spans[i - 1]["end"]:
                ok = False
                reasons.append(f"overlapping spans in {text!r}")
                break
        if not ok:
            continue

        ents = [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in seen_spans]
        valid.append({"text": text, "entities": ents})

    return valid, reasons


def _anthropic_usage(usage: anthropic.types.Usage) -> UsageStats:
    return UsageStats(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
    )


def generate_batch_anthropic(
    client: anthropic.Anthropic,
    system_blocks: list[dict],
    model_id: str,
    variation_key: str,
    n: int,
) -> tuple[list[dict], list[str], UsageStats]:
    variation = VARIATIONS[variation_key]
    user_msg = (
        f"Generate exactly {n} diverse Vietnamese clipboard examples.\n\n"
        f"Variation focus this batch: **{variation}**\n\n"
        f"Tuân thủ entity schema và xuất theo JSON schema đã ràng buộc."
    )
    response = client.messages.create(
        model=model_id,
        max_tokens=4096,
        system=system_blocks,
        messages=[{"role": "user", "content": user_msg}],
        output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    )
    response_text = next(b.text for b in response.content if b.type == "text")
    valid, reasons = parse_and_validate(response_text)
    return valid, reasons, _anthropic_usage(response.usage)


def generate_batch_openrouter(
    client: httpx.Client,
    system_text: str,
    model_id: str,
    variation_key: str,
    n: int,
) -> tuple[list[dict], list[str], UsageStats]:
    variation = VARIATIONS[variation_key]
    user_msg = (
        f"Generate exactly {n} diverse Vietnamese clipboard examples.\n\n"
        f"Variation focus this batch: **{variation}**\n\n"
        f"TuÃ¢n thá»§ entity schema vÃ  xuáº¥t theo JSON schema Ä‘Ã£ rÃ ng buá»™c. "
        f"Return only JSON."
        f' Exact shape: {{"examples":[{{"text":"...","entities":[{{"substring":"...","label":"PER"}}]}}]}}.'
        f" Labels must be only PER, ADDR, NOTE."
    )
    request_body = {
        "model": model_id,
        "max_tokens": 4096,
        "temperature": 0.8,
        "reasoning": {"effort": "none", "exclude": True},
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_msg},
        ],
        "response_format": {"type": "json_object"},
    }
    response = None
    for attempt in range(3):
        try:
            response = client.post(OPENROUTER_CHAT_URL, json=request_body)
            response.raise_for_status()
            break
        except httpx.HTTPError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    if response is None:
        raise RuntimeError("OpenRouter request did not return a response")
    payload = response.json()
    response_text = payload["choices"][0]["message"]["content"]
    valid, reasons = parse_and_validate(response_text)
    usage = payload.get("usage") or {}
    return (
        valid,
        reasons,
        UsageStats(
            input_tokens=usage.get("prompt_tokens", 0) or 0,
            output_tokens=usage.get("completion_tokens", 0) or 0,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", default="seed.jsonl")
    parser.add_argument("--output", default="synth_v1.jsonl")
    parser.add_argument("--env-file", default=".env", help="Dotenv file containing provider API keys")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openrouter"],
        default=os.getenv("LLM_PROVIDER", "anthropic"),
        help="LLM provider to use",
    )
    parser.add_argument("--model", default=None, help="Model id; defaults to provider-specific env/default")
    parser.add_argument("--total", type=int, default=200, help="Target total valid examples")
    parser.add_argument("--batch-size", type=int, default=10, help="Examples requested per Claude call")
    parser.add_argument("--seed-shots", type=int, default=12, help="Number of seed examples in few-shot block")
    parser.add_argument("--max-batches", type=int, default=None, help="Stop after this many API batches even if target is not reached")
    args = parser.parse_args()

    env_path = Path(args.env_file)
    if load_dotenv is not None:
        load_dotenv(env_path if env_path.exists() else None)

    if args.provider == "openrouter":
        api_key_name = "OPENROUTER_API_KEY"
        model_id = args.model or os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL_ID)
    else:
        api_key_name = "ANTHROPIC_API_KEY"
        model_id = args.model or os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL_ID)

    if not os.getenv(api_key_name):
        dotenv_hint = (
            f" Install python-dotenv or export {api_key_name} manually."
            if load_dotenv is None
            else f" Put {api_key_name} in {env_path}."
        )
        raise RuntimeError(f"{api_key_name} is not configured." + dotenv_hint)

    seed_block_text = _seed_block(Path(args.seed), n=args.seed_shots)

    # System blocks: SYSTEM_PROMPT + seed few-shot (cache breakpoint trên seed block).
    system_blocks = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "text", "text": seed_block_text, "cache_control": {"type": "ephemeral"}},
    ]
    system_text = f"{SYSTEM_PROMPT}\n\n{seed_block_text}"

    if args.provider == "openrouter":
        client = httpx.Client(
            timeout=120,
            headers={
                "Authorization": f"Bearer {os.environ[api_key_name]}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "task01-fix-data-generation"),
            },
        )
    else:
        client = anthropic.Anthropic()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    variations = list(VARIATIONS.keys())
    total_valid = 0
    total_rejected = 0
    total_cache_read = 0
    total_cache_write = 0
    total_input = 0
    total_output = 0
    timestamp = datetime.now(timezone.utc).isoformat()

    with open(output_path, "a", encoding="utf-8") as f:
        batch_idx = 0
        while total_valid < args.total:
            if args.max_batches is not None and batch_idx >= args.max_batches:
                print(
                    f"[stop] max_batches={args.max_batches} reached before target; "
                    f"valid={total_valid}/{args.total}",
                    file=sys.stderr,
                )
                break
            variation_key = variations[batch_idx % len(variations)]
            try:
                if args.provider == "openrouter":
                    valid, reasons, usage = generate_batch_openrouter(
                        client, system_text, model_id, variation_key, n=args.batch_size
                    )
                else:
                    valid, reasons, usage = generate_batch_anthropic(
                        client, system_blocks, model_id, variation_key, n=args.batch_size
                    )
            except anthropic.APIStatusError as e:
                print(f"[error] {e.status_code} {e.message}, skip batch", file=sys.stderr)
                batch_idx += 1
                continue
            except httpx.HTTPError as e:
                print(f"[error] OpenRouter request failed: {e}, skip batch", file=sys.stderr)
                batch_idx += 1
                continue
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"[error] response parse failed: {e}, skip batch", file=sys.stderr)
                batch_idx += 1
                continue

            for ex in valid:
                row = {
                    "id": f"synth-{uuid4().hex[:8]}",
                    "text": ex["text"],
                    "entities": ex["entities"],
                    "source": f"{args.provider}_synth",
                    "model": model_id,
                    "variation": variation_key,
                    "created_at": timestamp,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            total_valid += len(valid)
            total_rejected += len(reasons)
            total_cache_read += usage.cache_read_input_tokens or 0
            total_cache_write += usage.cache_creation_input_tokens or 0
            total_input += usage.input_tokens
            total_output += usage.output_tokens

            print(
                f"[batch {batch_idx:3d} {variation_key:22s}] "
                f"+{len(valid)}/{args.batch_size} valid, {len(reasons)} rej | "
                f"total={total_valid}/{args.total} | "
                f"cache_r={usage.cache_read_input_tokens or 0} cache_w={usage.cache_creation_input_tokens or 0}",
                flush=True,
            )
            batch_idx += 1

    total_attempted = total_valid + total_rejected
    reject_rate = (total_rejected / total_attempted) if total_attempted else 0.0
    print(
        f"\nDone. valid={total_valid}, rejected={total_rejected} ({reject_rate:.1%}), "
        f"batches={batch_idx}\n"
        f"Tokens: input={total_input}, output={total_output}, "
        f"cache_read={total_cache_read}, cache_write={total_cache_write}\n"
        f"Cost estimate (Anthropic Haiku 4.5 pricing, only valid for Anthropic): "
        f"${(total_input + total_cache_write*1.25 + total_cache_read*0.1) / 1e6 * 1.0 + total_output / 1e6 * 5.0:.4f}"
    )


if __name__ == "__main__":
    main()
