"""NER Service — ONNX Runtime CPU INT8 inference (v4).

Replaces the PyTorch AutoModelForTokenClassification backend with
onnxruntime InferenceSession.  No torch dependency at runtime.

Interface is backwards-compatible:
    NERService(model_path).is_ready() -> bool
    NERService(model_path).extract_tokens(text) -> List[TokenRow]
"""

import os
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

VALID_LABELS = {"O", "B-PER", "I-PER", "B-ADDR", "I-ADDR", "B-NOTE", "I-NOTE"}

# Supported ONNX artifact names, ordered by preference.
_ONNX_FILENAMES = ("model_quantized.onnx", "model.onnx")


def _resolve_onnx_path(model_path: str) -> Optional[str]:
    for filename in _ONNX_FILENAMES:
        candidate = os.path.join(model_path, filename)
        if os.path.exists(candidate):
            return candidate
    return None


@dataclass
class TokenRow:
    start: int
    end: int
    token: str
    label: str


class NERService:
    """ONNX-backed NER service for Vietnamese delivery-order parsing."""

    def __init__(self, model_path: str = "model/xlmr_ner_onnx_int8"):
        self.model_path = model_path
        self.tokenizer: Optional[AutoTokenizer] = None
        self.session: Optional[ort.InferenceSession] = None
        self.id2label: dict = {}
        self._has_native_offsets = False
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.model_path):
            logger.warning("NERService: model_path does not exist: %s", self.model_path)
            return

        onnx_path = _resolve_onnx_path(self.model_path)
        if onnx_path is None:
            logger.warning(
                "NERService: ONNX file not found in %s; expected one of: %s",
                self.model_path,
                ", ".join(_ONNX_FILENAMES),
            )
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            logger.info("NERService: tokenizer loaded from %s", self.model_path)
        except Exception as exc:
            logger.error("NERService: tokenizer load failed: %s", exc)
            return

        self._has_native_offsets = self._tokenizer_supports_offsets()
        if not self._has_native_offsets:
            logger.warning(
                "NERService: tokenizer from %s does not provide offset_mapping; "
                "using token-string offset fallback.",
                self.model_path,
            )

        try:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self.session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            logger.info("NERService: ONNX session loaded from %s", onnx_path)
        except Exception as exc:
            logger.error("NERService: ONNX session load failed: %s", exc)
            self.tokenizer = None
            return

        # Build id2label from config.json if available
        self._load_id2label()

    def _load_id2label(self) -> None:
        import json

        config_path = os.path.join(self.model_path, "config.json")
        if not os.path.exists(config_path):
            logger.warning("NERService: config.json not found, using default label map")
            self.id2label = {
                0: "O",
                1: "B-PER",
                2: "I-PER",
                3: "B-ADDR",
                4: "I-ADDR",
                5: "B-NOTE",
                6: "I-NOTE",
            }
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            raw = cfg.get("id2label", {})
            self.id2label = {int(k): v for k, v in raw.items()}
            logger.info("NERService: id2label loaded (%d labels)", len(self.id2label))
        except Exception as exc:
            logger.error("NERService: failed to parse config.json: %s", exc)
            self.id2label = {}

    def _tokenizer_supports_offsets(self) -> bool:
        if self.tokenizer is None:
            return False
        try:
            encoding = self.tokenizer(
                "offset check",
                return_offsets_mapping=True,
                return_tensors="np",
                truncation=True,
                max_length=16,
                padding=False,
            )
        except Exception as exc:
            logger.error("NERService: tokenizer offset check failed: %s", exc)
            return False
        return "offset_mapping" in encoding

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        if self.tokenizer is None or self.session is None:
            return False
        if not self.id2label:
            return False
        for label in self.id2label.values():
            if label not in VALID_LABELS:
                return False
        return True

    def extract_tokens(self, text: str) -> List[TokenRow]:
        """Run ONNX inference and return per-token predictions."""
        if not text:
            return []

        tokenizer_kwargs = {
            "return_tensors": "np",
            "truncation": True,
            "max_length": 512,
            "padding": False,
        }
        has_native_offsets = getattr(self, "_has_native_offsets", True)
        if has_native_offsets:
            tokenizer_kwargs["return_offsets_mapping"] = True

        encoding = self.tokenizer(text, **tokenizer_kwargs)

        input_ids: np.ndarray = encoding["input_ids"]          # (1, seq)
        attention_mask: np.ndarray = encoding["attention_mask"] # (1, seq)
        token_ids = input_ids[0].tolist()
        tokens = [self.tokenizer.convert_ids_to_tokens(int(token_id)) for token_id in token_ids]
        if has_native_offsets:
            offsets = encoding["offset_mapping"][0].tolist()    # list of [s, e]
        else:
            offsets = self._build_offsets_from_tokens(text, tokens)

        # Build ONNX input dict — only pass inputs the model expects
        ort_inputs = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
        }

        # Some models also expect token_type_ids
        input_names = {inp.name for inp in self.session.get_inputs()}
        if "token_type_ids" in input_names:
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
            ort_inputs["token_type_ids"] = token_type_ids

        logits = self.session.run(None, ort_inputs)[0]  # (1, seq, num_labels)
        predictions = np.argmax(logits[0], axis=-1).tolist()

        token_rows: List[TokenRow] = []
        for idx, (offset, pred_id) in enumerate(zip(offsets, predictions)):
            if offset == [0, 0] or offset == (0, 0):
                continue  # skip special tokens

            token_str = tokens[idx]
            label = self.id2label.get(pred_id, "O")

            token_rows.append(
                TokenRow(
                    start=offset[0],
                    end=offset[1],
                    token=token_str,
                    label=label,
                )
            )

        return token_rows

    def _build_offsets_from_tokens(self, text: str, tokens: List[str]) -> List[List[int]]:
        """Best-effort offsets for slow BPE tokenizers such as PhoBERT."""
        offsets: List[List[int]] = []
        cursor = 0
        specials = set(getattr(self.tokenizer, "all_special_tokens", []) or [])

        for token in tokens:
            if token in specials:
                offsets.append([0, 0])
                continue

            piece = self._token_to_text_piece(token)
            if not piece:
                offsets.append([0, 0])
                continue

            while cursor < len(text) and text[cursor].isspace():
                cursor += 1

            start = text.find(piece, cursor)
            if start < 0:
                # Last-resort fallback keeps spans monotonic instead of crashing.
                start = min(cursor, len(text))
                end = min(len(text), start + len(piece))
            else:
                end = start + len(piece)

            offsets.append([start, end])
            cursor = end

        return offsets

    def _token_to_text_piece(self, token: str) -> str:
        if token.startswith("▁"):
            token = token[1:]
        if token.endswith("@@"):
            token = token[:-2]
        return token
