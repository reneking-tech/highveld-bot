Highveld Biotech Assistant

A small retrieval‑augmented chatbot. FastAPI backend with JSON/streaming endpoints, a minimal web UI, and a PDF quote generator. The knowledge base is a CSV embedded into a FAISS index.

Purpose
- Answer questions about lab tests, pricing, turnaround times, and drop‑off details using a small, local RAG stack.
- Provide simple endpoints for chat and streaming, plus a downloadable quote PDF for hand‑off to sales.

Tech stack
- Python 3.11+
- FastAPI, Uvicorn
- LangChain (chat + retriever), langchain‑openai, OpenAI Chat/Embeddings
- FAISS (vector index), pandas (CSV ingest)
- ReportLab (PDF)
- pytest (tests), python‑dotenv (env)

Key features
- RAG over CSV with FAISS and MMR retriever
- Guardrails: address override, keyword‑overlap + vector‑distance gating, “not sure” fallback, price guard
- Streaming: /stream (text) and /stream/sse (SSE)
- Quote PDF generation endpoint
- Optional conversation memory (SQLite)
- Prompt versioning and simple per‑IP rate limiting
- Unit tests for core behaviors; eval scaffolding present

Project structure
- run_api.py
- api/server.py
- app/ingest.py, app/retriever.py, app/rag_chain.py, app/prompts.py, app/memory.py, app/pdf.py, app/config.py
- data/lab_faq.csv
- web/index.html, web/styles.css
- tests/
- requirements.txt, pyproject.toml
- .env.example

Requirements
- Python 3.11+
- OPENAI_API_KEY in .env

Setup
1) Create a venv and install
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
2) Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

Ingest data
- `python -m app.ingest`

Run API
- `python run_api.py`
- Health: GET `/healthz`
- UI: http://127.0.0.1:8000/

Example usage
- Chat
```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"How much is a full SANS 241 test?"}'
```

- Streaming (plain text chunks)
```bash
curl -N -s -X POST http://127.0.0.1:8000/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"Summarise Core Water Analysis"}'
```

- Quote PDF
```bash
curl -s -X POST http://127.0.0.1:8000/quote/pdf \
  -H 'Content-Type: application/json' \
  -d '{"message":"Quote for Core Water Analysis and Full SANS 241"}' \
  -o quote.pdf
```

Endpoints
- POST `/chat`
- POST `/stream`
- POST `/stream/sse`
- POST `/quote/pdf`
- POST `/twilio/whatsapp`
- GET `/healthz`

Tests
- `python -m pytest -q`
