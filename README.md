Highveld Biotech Assistant

A small retrieval‑augmented chatbot. FastAPI backend with JSON/streaming endpoints, a minimal web UI, and a PDF quote generator. The knowledge base is a CSV embedded into a FAISS index.

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

Endpoints
- POST `/chat`
- POST `/stream`
- POST `/stream/sse`
- POST `/quote/pdf`
- POST `/twilio/whatsapp`
- GET `/healthz`

Tests
- `python -m pytest -q`
