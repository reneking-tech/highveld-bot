"""
FastAPI server for the Highveld Biotech chatbot.
Serves the web UI and provides API endpoints for RAG interactions.
"""

from app.memory import MemoryService
from app.state import ChatSession
from app.rag_chain import RAGService
from app.pdf import generate_quote_pdf
import argparse
import os
import socket
from contextlib import closing
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load environment variables early
load_dotenv(find_dotenv())

# Add the project root to PYTHONPATH for imports
sys.path.insert(0, str(Path(__file__).parent))


def _port_available(host: str, port: int) -> bool:
    """Return True if we can bind to the given host:port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _fallback_app():
    """Minimal fallback app if api.server.app import fails."""
    from fastapi import FastAPI

    app = FastAPI(title="Highveld Biotech Assistant (fallback)")

    @app.get("/")
    def root():
        return {
            "ok": True,
            "message": "API fallback is running. Could not import api.server.app.",
            "hint": "Check your virtualenv and PYTHONPATH. Open /static/index.html if static files are mounted by the main app.",
        }

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app


app = FastAPI(title="Highveld Biotech Chatbot", version="1.0.0")

# Mount static files (web UI)
app.mount("/static", StaticFiles(directory="web"), name="static")


class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class QuotePDFRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    # Minimal client fields for PDF header
    name: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None
    phone: Optional[str] = None
    reference: Optional[str] = None
    extra: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web UI."""
    return FileResponse("web/index.html")


@app.get("/static/index.html", response_class=HTMLResponse)
async def static_index():
    """Serve the web UI at /static/index.html for compatibility."""
    return FileResponse("web/index.html")


@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    """Endpoint for asking questions to the chatbot."""
    try:
        service = RAGService()
        memory = MemoryService(session_id=request.session_id)
        session = memory.load_session() or ChatSession()

        result = service.answer_question(request.question, session=session)

        # Save session if updated
        if session.selected_tests or session.state != "idle":
            memory.save_session(session)

        # Augment response with a readiness hint for the UI
        try:
            ready_for_pdf = (
                getattr(session, "state", "") == "pdf"
                and bool(getattr(session, "client_name", ""))
                and bool(getattr(session, "client_email", ""))
                and bool(getattr(session, "client_phone", ""))
                and bool(getattr(session, "selected_tests", []))
            )
        except Exception:
            ready_for_pdf = False
        if isinstance(result, dict):
            result.setdefault("meta", {})
            result["meta"]["ready_for_pdf"] = ready_for_pdf
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/structured-quote")
async def get_structured_quote(request: QuestionRequest):
    """Endpoint for structured quote requests."""
    try:
        service = RAGService()
        result = service.structured_quote(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quote/pdf")
async def get_quote_pdf(request: QuotePDFRequest):
    """Generate and return a PDF from a structured quote.

    Accepts a natural-language question to derive the structured quote
    and optional client details for the PDF header.
    """
    try:
        service = RAGService()
        obj = service.structured_quote(request.question)

        tests = obj.get("tests") or []
        items = []
        # Prefer session-selected tests if available
        sess_for_items = None
        if request.session_id:
            try:
                mem_i = MemoryService(session_id=request.session_id)
                sess_for_items = mem_i.load_session()
            except Exception:
                sess_for_items = None
        if sess_for_items and getattr(sess_for_items, "selected_tests", []):
            for t in sess_for_items.selected_tests:
                qty = 1
                try:
                    sc = getattr(sess_for_items, "sample_counts", {}) or {}
                    if t.code in sc:
                        qty = max(1, int(sc[t.code]))
                except Exception:
                    pass
                items.append(
                    {
                        "name": t.name,
                        "price_ZAR": t.price,
                        "turnaround_days": getattr(t, "tat_days", 0),
                        "quantity": qty,
                    }
                )
        else:
            # Fall back to structured_quote items
            for t in tests:
                items.append(
                    {
                        "name": t.get("name", "Test"),
                        "price_ZAR": t.get("price_ZAR"),
                        "turnaround_days": t.get("turnaround_days"),
                        "quantity": int(t.get("quantity", 1) or 1),
                    }
                )

        # If details are not provided in the request, try load them from the session
        sess_client = {"name": "", "email": "", "company": "", "phone": "", "reference": ""}
        if request.session_id:
            try:
                mem = MemoryService(session_id=request.session_id)
                sess = mem.load_session()
                if sess:
                    sess_client = {
                        "name": (
                            (
                                getattr(sess, "client_name", "")
                                + " "
                                + getattr(sess, "client_surname", "")
                            ).strip()
                        ),
                        "email": getattr(sess, "client_email", ""),
                        "company": getattr(sess, "client_company", ""),
                        "phone": getattr(sess, "client_phone", ""),
                        "reference": "",
                    }
            except Exception:
                pass

        client = {
            "name": (request.name or sess_client["name"] or ""),
            "email": (request.email or sess_client["email"] or ""),
            "company": (request.company or sess_client["company"] or ""),
            "phone": (request.phone or sess_client["phone"] or ""),
            "reference": (request.reference or sess_client["reference"] or ""),
        }

        total = float(obj.get("total_price_ZAR") or 0.0)
        # Merge backend notes with client-provided extra info and standard policy/banking
        notes = (obj.get("notes", "") or "").strip()
        if request.extra:
            notes = (notes + "\n\nClient notes: " + request.extra.strip()).strip()
        try:
            from app import config as _cfg

            bank = getattr(_cfg, "BANK_DETAILS", "")
            policy = getattr(_cfg, "PAYMENT_POLICY", "")
            validity_days = int(getattr(_cfg, "QUOTE_VALIDITY_DAYS", 14))
            rush_threshold = float(getattr(_cfg, "RUSH_FEE_THRESHOLD_ZAR", 10000.0))
            rush_rate = float(getattr(_cfg, "RUSH_FEE_RATE", 0.30))
        except Exception:
            bank = policy = ""
            validity_days = 14
            rush_threshold, rush_rate = 10000.0, 0.30
        import datetime as _dt

        valid_until = (_dt.datetime.utcnow() + _dt.timedelta(days=validity_days)).strftime(
            "%Y-%m-%d"
        )
        if total >= rush_threshold:
            notes += f"\n\nRush option available: +{int(rush_rate*100)}% for results within 48 hours (eligible on orders above R{rush_threshold:,.0f})."
        if bank:
            notes = (notes + f"\n\n{bank}").strip()
        if policy:
            notes = (notes + f"\n\n{policy}").strip()
        msg = "Quote for requested laboratory tests"

        qref = f"HVB-{_dt.datetime.utcnow().strftime('%Y%m%d')}-{(request.session_id or 'Q')[-4:]}"
        pdf_bytes = generate_quote_pdf(
            client=client,
            items=items,
            message=msg,
            notes=notes,
            total_price_ZAR=total,
            quote_ref=qref,
            valid_until=valid_until,
        )

        from fastapi import Response
        from datetime import datetime

        fname = f"quote_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

        # Record the event in memory and gently reset flow to idle
        try:
            if request.session_id:
                memlog = MemoryService(session_id=request.session_id)
                sesslog = memlog.load_session() or ChatSession()
                memlog.save_turn(
                    "generate pdf quote", f"Generated PDF quote {qref} (total R{total:,.2f})"
                )
                # Reset conversational state to invite new questions
                sesslog.state = "idle"
                memlog.save_session(sesslog)
        except Exception:
            pass

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={fname}",
                "X-Quote-Ref": qref,
                "X-Quote-Valid-Until": valid_until,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Highveld Biotech Assistant API")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.getenv("RELOAD", "0").strip().lower() not in {"0", "false", "no"},
        help="Enable auto-reload (dev only)",
    )
    args = parser.parse_args()

    host = args.host
    port = args.port
    if not _port_available(host, port):
        print(f"[run_api] Port {port} is busy; selecting an ephemeral port.")
        port = 0

    # When reload is enabled, Uvicorn requires an import string, not an app object
    # Configure reload to ignore virtualenv and heavy data dirs to prevent loops
    uvicorn_kwargs = dict(
        host=host,
        port=port,
        reload=args.reload,
        log_level="info",
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
    if args.reload:
        base = Path(__file__).parent
        # Limit what we watch and exclude chatty directories like the venv
        uvicorn_kwargs.update(
            {
                "reload_dirs": [str(base), str(base / "app"), str(base / "api"), str(base / "web")],
                "reload_excludes": [
                    ".venv/*",
                    "venv/*",
                    "env/*",
                    "**/.venv/*",
                    "**/venv/*",
                    "**/__pycache__/*",
                    "**/*.pyc",
                    ".git/*",
                    ".mypy_cache/*",
                    "data/*",
                    ".data/*",
                    ".vectordb/*",
                ],
            }
        )
        uvicorn.run("run_api:app", **uvicorn_kwargs)
    else:
        uvicorn.run(app, **uvicorn_kwargs)


if __name__ == "__main__":
    main()
