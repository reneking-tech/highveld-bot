from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

from app.rag_chain import RAGService
from app.memory import MemoryService
from app.state import ChatSession

# try to import our shared error model
try:
    from api.models import ErrorEnvelope
except ImportError:
    from .models import ErrorEnvelope  # fallback if used as a package

# Load environment (.env optional)
load_dotenv(find_dotenv())

APP_TITLE = os.getenv("APP_TITLE", "Highveld Bot API")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# CORS (wide-open by default; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_session_id(sid: Optional[str]) -> str:
    return sid or str(uuid.uuid4())


# ------------ Models ------------


class ChatPayload(BaseModel):
    q: str
    session_id: Optional[str] = None
    extra_context: Optional[str] = None


class QuotePayload(BaseModel):
    q: str
    session_id: Optional[str] = None


# ------------ Routes ------------


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": APP_TITLE, "version": APP_VERSION}


@app.post("/chat")
def chat(payload: ChatPayload):
    q = (payload.q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing 'q'")

    # Use per-session memory and RAG service to avoid global init issues
    session_id = _ensure_session_id(payload.session_id)
    service = RAGService(session_id=session_id)
    memory = MemoryService(session_id=session_id)

    # Load existing session state if available
    sess = memory.load_session() or ChatSession(session_id=session_id)
    ans = service.answer_question(
        q, extra_context=payload.extra_context, session=sess)

    # Persist updated session state
    try:
        memory.save_session(sess)
    except Exception:
        # Non-fatal if persistence fails
        pass

    return ans


@app.post("/quote/structured")
def quote_structured(payload: QuotePayload):
    q = (payload.q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing 'q'")

    session_id = _ensure_session_id(payload.session_id)
    service = RAGService(session_id=session_id)
    obj = service.structured_quote(q)
    return obj


@app.get("/")
def root():
    return {"message": "Highveld Bot API. See /health, POST /chat, POST /quote/structured"}


# ------------ Exception Handlers ------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    env = ErrorEnvelope(
        code=str(exc.status_code),
        message=str(exc.detail or "HTTP error"),
        details={"path": str(request.url)},
    )
    return JSONResponse(status_code=exc.status_code, content=env.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    env = ErrorEnvelope(
        code="internal_error",
        message="Unexpected server error",
        details={"path": str(request.url)},
    )
    return JSONResponse(status_code=500, content=env.model_dump())
