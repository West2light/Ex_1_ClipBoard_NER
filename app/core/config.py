"""Application settings (M3 — dual-model configuration).

All tuneable values are read from environment variables.  The module
exposes a module-level singleton ``settings`` used across the app.
"""

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a runtime dependency here
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

_SUPPORTED_FAMILIES = {"xlmr", "phobert"}

_DEFAULT_MODEL_PATHS = {
    "xlmr": "model/xlmr_ner_onnx_int8",
    "phobert": "model/phobert_ner_onnx_int8",
}


def _env_value(name: str, default: str = "") -> str:
    value = os.environ.get(name, default)
    return value.split("#", 1)[0].strip()


@dataclass
class Settings:
    # Model selection
    MODEL_FAMILY: str = field(
        default_factory=lambda: _env_value("MODEL_FAMILY", "xlmr").lower()
    )
    # Explicit override; if empty, resolved from MODEL_FAMILY
    MODEL_PATH: str = field(
        default_factory=lambda: _env_value("MODEL_PATH", "")
    )

    # Feature flags
    DEBUG_ENABLED: bool = field(
        default_factory=lambda: _env_value("DEBUG_ENABLED", "").lower()
        in ("1", "true", "yes")
    )

    # Limits
    MAX_INPUT_LENGTH: int = field(
        default_factory=lambda: int(_env_value("MAX_INPUT_LENGTH", "5000"))
    )

    # Logging
    LOG_LEVEL: str = field(
        default_factory=lambda: _env_value("LOG_LEVEL", "INFO").upper()
    )

    def __post_init__(self) -> None:
        if self.MODEL_FAMILY not in _SUPPORTED_FAMILIES:
            raise ValueError(
                f"Unsupported MODEL_FAMILY={self.MODEL_FAMILY!r}. "
                f"Choose from: {sorted(_SUPPORTED_FAMILIES)}"
            )

    @property
    def resolved_model_path(self) -> str:
        """Return the absolute-or-relative model directory path."""
        if self.MODEL_PATH:
            return self.MODEL_PATH
        return _DEFAULT_MODEL_PATHS[self.MODEL_FAMILY]

    @property
    def use_phobert(self) -> bool:
        return self.MODEL_FAMILY == "phobert"


# Module-level singleton — imported by the rest of the app.
settings = Settings()
