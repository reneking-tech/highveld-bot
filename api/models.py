from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message/question")
    session_id: Optional[str] = Field(None, description="Opaque session identifier")


class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None
    low_confidence: bool = False
    options: Optional[List[str]] = None


class QuoteRequest(BaseModel):
    name: str
    email: EmailStr
    phone: str
    tests: List[str]
    notes: Optional[str] = None


class ErrorEnvelope(BaseModel):
    code: str
    message: str
    details: Optional[dict] = None
