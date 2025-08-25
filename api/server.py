"""FastAPI server exposing chat, streaming, WhatsApp webhook, and PDF quote.

Includes simple per-IP rate limiting and serves a minimal static web UI for
quick manual testing.
"""

from __future__ import annotations
from app.rag_chain import RAGService

import os
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


app = FastAPI(title="Highveld Bot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve a simple web chat UI for quick testing
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
WEB_DIR = os.path.join(ROOT_DIR, "web")
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    # Also expose repo root as /assets so we can reference /assets/logo.png
    app.mount("/assets", StaticFiles(directory=ROOT_DIR), name="assets")

# Simple rate limiting (per-IP per minute)
try:
    from app.config import RATE_LIMIT_PER_MIN  # type: ignore
except Exception:
    RATE_LIMIT_PER_MIN = 60

_rate_bucket: dict[str, tuple[int, float]] = {}


def _check_rate_limit(ip: str) -> None:
    import time
    now = time.time()
    count, reset = _rate_bucket.get(ip, (0, now + 60))
    if now > reset:
        count, reset = 0, now + 60
    count += 1
    _rate_bucket[ip] = (count, reset)
    if count > int(RATE_LIMIT_PER_MIN):
        raise HTTPException(status_code=429, detail="Too Many Requests. Please slow down.")


@app.get("/")
def root_page():
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"status": "ok", "message": "API is running"}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    context: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str


class ClientDetails(BaseModel):
    name: str | None = None
    company: str | None = None
    email: str | None = None
    phone: str | None = None
    reference: str | None = None
    billing_address: str | None = None
    vat_no: str | None = None


class QuotePdfRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[str] = None
    client: Optional[ClientDetails] = None


"""Simple in-memory session store (per-process).
We store RAGService instances per session so each retains its own memory.
"""
_sessions: Dict[str, RAGService] = {}


def _get_service(session_id: Optional[str], *, model: Optional[str], temperature: Optional[float], top_k: Optional[int]) -> tuple[str, RAGService]:
    sid = session_id or str(uuid.uuid4())
    svc = _sessions.get(sid)
    if svc is None:
        svc = RAGService(llm_model=model, temperature=temperature, top_k=top_k, session_id=sid)
        _sessions[sid] = svc
    return sid, svc


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    _check_rate_limit(str(request.client.host) if request.client else "chat")
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")
    sid, rag = _get_service(req.session_id, model=req.model,
                            temperature=req.temperature, top_k=req.top_k)
    result = rag.answer_question(req.message, extra_context=req.context)
    return ChatResponse(answer=result["answer"], sources=result.get("sources", []), session_id=sid)


@app.post("/stream")
def stream(req: ChatRequest, request: Request):
    _check_rate_limit(str(request.client.host) if request.client else "stream")
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")
    sid, rag = _get_service(req.session_id, model=req.model,
                            temperature=req.temperature, top_k=req.top_k)

    def gen():
        for chunk in rag.stream_answer(req.message, extra_context=req.context):
            yield chunk

    return StreamingResponse(gen(), media_type="text/plain")


@app.post("/stream/sse")
def stream_sse(req: ChatRequest, request: Request):
    _check_rate_limit(str(request.client.host) if request.client else "sse")
    _check_rate_limit(req.session_id or "sse:" + "0")
    """Server-Sent Events streaming endpoint returning text/event-stream.
    Each chunk is sent as an SSE data event.
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")
    sid, rag = _get_service(req.session_id, model=req.model,
                            temperature=req.temperature, top_k=req.top_k)

    def gen():
        try:
            for chunk in rag.stream_answer(req.message, extra_context=req.context):
                yield f"data: {chunk}\n\n"
            yield "event: end\ndata: [DONE]\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


# Minimal Twilio WhatsApp webhook
@app.post("/twilio/whatsapp")
async def twilio_whatsapp(request: Request):
    _check_rate_limit(str(request.client.host) if request.client else "twilio")
    # Twilio sends application/x-www-form-urlencoded
    form = await request.form()
    body = str(form.get("Body", "")).strip()
    from_ = form.get("From") or ""
    # Use the WhatsApp phone as the session id
    sid, rag = _get_service(from_ or None, model=None,
                            temperature=None, top_k=None)
    result = rag.answer_question(body or "")
    # Respond in TwiML so Twilio replies back to the user
    reply = result.get("answer", "")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Message>{reply}</Message>
</Response>"""
    return PlainTextResponse(content=twiml, media_type="application/xml")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/quote/pdf")
def quote_pdf(req: QuotePdfRequest, request: Request):
    _check_rate_limit(str(request.client.host) if request.client else "pdf")
    _check_rate_limit(req.session_id or "pdf:" + "0")
    """Return a generated PDF for a given quote question.
    Uses the structured_quote path to assemble items and totals, then renders a PDF.
    """
    from app.pdf import render_quote_pdf
    sid, rag = _get_service(req.session_id, model=None,
                            temperature=None, top_k=None)
    # Build structured quote from the question and optional context
    question = req.message
    if req.context:
        question = f"{req.message}\n\n{req.context}"
    quote = rag.structured_quote(question)
    # Convert Pydantic model to dict for client block
    client_dict = req.client.dict() if req.client else {}
    pdf_bytes = render_quote_pdf(quote, client=client_dict)
    headers = {"Content-Disposition": "attachment; filename=highveld_quote.pdf"}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
