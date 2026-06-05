"""Pydantic request/response schemas (M4) per api_contract_v1.md."""

from typing import Optional
from pydantic import BaseModel, ConfigDict


class ParseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    debug: bool = False


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
    debug: Optional[dict] = None


class ParseResponse(BaseModel):
    recipient_name: Optional[str] = None
    phone_number: Optional[str] = None
    address_raw: str
    note: Optional[str] = None
    debug: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    status: str
    model_family: str
