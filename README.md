# Highveld Biotech Assistant

A small retrieval‑augmented chatbot. FastAPI backend with JSON/streaming endpoints, a minimal web UI, and a PDF quote generator. The knowledge base is a CSV embedded into a FAISS index.

## Purpose
- Answer questions about lab tests, pricing, turnaround times, and drop‑off details using a small, local RAG stack.
- Provide simple endpoints for chat and streaming, plus a downloadable quote PDF for hand‑off to sales.

## Tech stack
- Python 3.11+
- FastAPI, Uvicorn
- LangChain (chat + retriever), langchain‑openai, OpenAI Chat/Embeddings
- FAISS (vector index), pandas (CSV ingest)
- ReportLab (PDF)
- pytest (tests), python‑dotenv (env)

## Key features
- RAG over CSV with FAISS and MMR retriever
- Guardrails: address override, keyword‑overlap + vector‑distance gating, “not sure” fallback, price guard
- Streaming: /stream (text) and /stream/sse (SSE)
- Quote PDF generation endpoint
- Optional conversation memory (SQLite)
- Prompt versioning and simple per‑IP rate limiting
- Unit tests for core behaviors; eval scaffolding present

## Project structure
- run_api.py
- api/server.py
- app/ingest.py, app/retriever.py, app/rag_chain.py, app/prompts.py, app/memory.py, app/pdf.py, app/config.py
- data/lab_faq.csv
- web/index.html, web/styles.css
- tests/
- requirements.txt, pyproject.toml
- .env.example

## Requirements
- Python 3.11+
- OPENAI_API_KEY in .env

## Setup
1) Create a venv and install
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
2) Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

## Ingest data
- `python -m app.ingest`

## Run API
- `python run_api.py`
- Health: GET `/healthz`
- UI: http://127.0.0.1:8000/

## Example usage
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

## Endpoints
- POST `/chat`
- POST `/stream`
- POST `/stream/sse`
- POST `/quote/pdf`
- POST `/twilio/whatsapp`
- GET `/healthz`

## Tests
- `python -m pytest -q`

# Highveld Biotech Chatbot

Quick start
- python3 -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt
- Copy .env and set OPENAI_API_KEY
- Build index: python app/build_all.py
- Run API: python run_api.py
- Open http://127.0.0.1:8000

Quote flow
- Ask for pricing. If ambiguous, the bot lists options to confirm.
- Once a single test is confirmed, the bot replies with “Quote:” and the web form appears.
- Fill in required fields (name, company, email, phone) and click Download PDF to get an instant quote file.

Endpoints
- POST /chat  → plain English by default; uses sticky cookie "sid"
- POST /chat/json → JSON {answer, sources?, session_id}
- POST /stream, /stream/sse → streaming
- POST /quote/pdf → returns a PDF
- GET /healthz → health check

Rebuild index when:
- data/ changes, EMBEDDING_MODEL changes, or chunking changes.

Memory
- ConversationSummaryBufferMemory with optional SQLite persistence (.data/memory.sqlite3)
- The service summarizes long histories so follow‑ups are contextual.

Guiding principles (condensed)
- Accuracy & trust: only answer from validated data; defer if unsure.
- Context & memory: retain multi‑turn context and summarize long threads.
- Professional tone: concise, warm, action‑oriented; offer next steps.
- Structure: scannable bullets; quotes rendered cleanly when asked.
- Scalability: modular ingestion/retrieval; environment-driven config.
- Human escalation: cleanly hand off when sensitive or ambiguous.
- Efficiency: sub‑second retrieval; small embeddings; tuned top_k.
- Evaluation: add feedback hooks and track answer quality over time.

Dev
- Lint: ruff check .
- Typecheck: mypy .
- Tests: pytest -q

# Highveld Biotech Bot — Deploy & Test

## 1) Prerequisites
- Python 3.11+
- A valid OpenAI API key in .env (do not commit)
  - OPENAI_API_KEY=sk-...
  - Optionally set OPENAI_MODEL=gpt-4o-mini and EMBEDDING_MODEL=text-embedding-3-small

## 2) Install
pip install -r requirements.txt

## 3) Build the index (one-time or when data changes)
python -m app.build_all

## 4) Run the API
python run_api.py
# Note the URL (e.g., http://127.0.0.1:8000)

## 5) Open the web UI
Local:
- cd web
- python -m http.server 8080
- Open http://localhost:8080/index.html?api=http://127.0.0.1:8000

Shareable (quick): expose your API
- ngrok http 8000
- Copy the ngrok URL (e.g., https://abcd1234.ngrok.io)
- Open your local UI with the API param:
  http://localhost:8080/index.html?api=https://abcd1234.ngrok.io

## 6) Can I keep making changes?
Yes.
- Backend: edit Python files, Ctrl+C to stop the API, then `python run_api.py` again.
- Frontend: edit web/*.html, *.css and refresh the browser.
- If you change embeddings/data, re-run: `python -m app.build_all`.
- If Joe is using your public API, restart the API on that host (or re-deploy) and the UI keeps working with the same ?api= URL.

## 7) Troubleshooting
- OPENAI_BASE_URL unset or invalid → leave it blank in .env or set to a full URL (https://api.openai.com/v1).
- Vector index missing → run `python -m app.build_all`.
- CORS with remote API → serve UI and API on the same origin, or enable CORS in your FastAPI app.

# Highveld Biotech Assistant — Quick Start

Setup
- Python 3.11+
- cp .env.example .env (ensure OPENAI_API_KEY present)
- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt

Build index
- python -m app.build_all

Run
- python run_api.py
- Open http://127.0.0.1:8000/static/index.html

# Highveld Biotech Assistant — System Overview

What it is
- A FastAPI-based RAG assistant that helps clients select tests, get quotes, generate formal PDF quotes, and handle logistics (sample prep, drop‑off).
- Uses a FAISS vector index over curated test data; prompts an OpenAI chat model; maintains a compact rolling summary memory.
- Ships with a minimal web UI, solid guardrails (price integrity), and a stateful quote flow.

High-level architecture
- API server (api/server.py)
  - Endpoints: /chat, /chat/json, /stream, /stream/sse, /quote/plan, /quote/pdf, /twilio/whatsapp, /session/reset, /healthz
  - Simple per-IP rate limiting; session reuse via cookie or provided session_id
- RAG service (app/rag_chain.py)
  - Retrieval: langchain FAISS retriever (MMR) over compiled catalog
  - LLM: ChatOpenAI with low temperature for factuality
  - Memory: ConversationSummaryBufferMemory wrapper (app/memory.py)
  - Stateful quote flow with ChatSession (app/state.py)
  - PDF generation with reportlab (app/pdf.py)
- Data and indexing
  - Prepare: app/prepare_embeddings.py cleans/normalizes sources into data/compiled/*
  - Ingest: app/ingest.py builds FAISS index at .vectordb/faq_faiss
  - Retriever: app/retriever.py loads FAISS and exposes an as_retriever(k)
- Prompting
  - app/prompts.py provides system/user templates and loads principles.txt, quote_principles.txt
- Configuration
  - app/config.py centralizes env, paths, knobs; sanitizes OPENAI_BASE_URL

Core flows (behavior)
- General Q&A (default)
  - Retrieve top-k chunks; format a compact context; answer succinctly with a professional tone
  - Confidence gates: keyword-overlap + distance; returns a helpful clarification if low-confidence (no invention)
  - Location-only queries: respond with canonical drop‑off address
- Quote flow (Discovery → Selection → Selection_confirm → Confirmation → PDF)
  - Discovery: broad ask → show 3–5 relevant options (name, price, TAT); no totals
  - Selection: accept numbers/names; confirm additions; offer to add more
  - Confirmation: itemized draft with total + 14‑day validity; ask to generate formal PDF
  - PDF: collect First name → Surname → Company → Email; generate and save web/quotes/Quote-<ref>.pdf; reply with download link; reset session
  - Late additions during PDF: pause, return to selection, then reconfirm
- Sample collection questions (e.g., “how do I collect a water sample”)
  - Direct, practical guidance (container, headspace, chilling, timing, drop‑off)
- Escalation (human handover)
  - Detect “speak to a human/consultant/call me”; pause flows; collect name + contact; handover path (hook point for CRM/inbox)

Retrieval and data
- Source of truth: data/compiled/lab_tests_clean.{json,csv} (id, name, price_ZAR, turnaround_days, category, notes, standard, kind, embedding_text)
- Prepare (one source of truth):
  - python -m app.prepare_embeddings → writes compiled JSON/CSV
- Ingest/build:
  - python -m app.ingest → FAISS index under .vectordb/faq_faiss
- Runtime:
  - retriever = vectordb.as_retriever(search_type="mmr", k=TOP_K)

Prompting and principles
- app/prompts.py: system/user templates for both RAG prose and structured quote JSON
- principles.txt and quote_principles.txt are injected as additional system messages
- Style: concise, professional; no IDs/JSON in prose; prices in ZAR; end with a clear next step

Memory and persistence
- app/memory.py wraps ConversationSummaryBufferMemory with version-safe init (llm vs summary_llm)
- Stores summary under “history”; optional SQLite persistence per session (ENABLE_PERSISTENT_MEMORY, MEMORY_DB_PATH)
- Memory used to keep continuity and reduce repetition while not overriding retrieved facts

API surface (FastAPI)
- POST /chat → {"answer": string, "session_id": string}
- POST /chat/json → ChatResponse (answer, sources[], session_id)
- POST /stream → chunked plain text
- POST /stream/sse → event-stream chunks
- POST /quote/plan → structured JSON plan (options or confirmed)
- POST /quote/pdf → returns PDF as application/pdf (validated inputs, catalog-matched items only)
- POST /twilio/whatsapp → TwiML response for WhatsApp inbound
- POST /session/reset → clear server-side session, best-effort memory wipe
- GET /healthz → health check
- GET /static/* → minimal web UI and saved PDFs

Configuration (app/config.py, .env)
- OPENAI_API_KEY, OPENAI_BASE_URL (sanitized), OPENAI_MODEL/LLM_MODEL
- EMBEDDING_MODEL, RAG_TOP_K, CHUNK_SIZE/OVERLAP
- INDEX_DIR, PROJECT_ROOT, DATA_FILE (auto-fallbacks)
- MIN_KEYWORD_OVERLAP, SIMILARITY_DISTANCE_MAX, ENABLE_NOT_SURE_FALLBACK
- ENABLE_PRICE_GUARD
- ENABLE_PERSISTENT_MEMORY, MEMORY_DB_PATH
- RATE_LIMIT_PER_MIN

Guardrails and safety
- Price integrity guard: only echo prices present in retrieved context; fallback if absent
- Address/drop‑off cards filtered from quote options
- Clarify rather than guess when ambiguous; smooth escalation offer
- Professional tone guidelines via principles and style guide
- No medical advice

Dev quick start
- Build index: python -m app.build_all
- Run API: python run_api.py → http://127.0.0.1:8000/static/index.html
- Tests: pytest (tests/ directory)
- Env: set OPENAI_API_KEY in .env or shell; base URL sanitized automatically

Agentic development hooks (where to extend)
- Tools
  - Pricing/ERP lookup tool: authoritative prices, stock, lead times
  - CRM/logging tool: persist quotes, escalation requests, outcomes
  - Email/SMS tool: send PDF quotes and confirmations
- Self-reflection
  - Post-answer heuristic scoring (helpfulness, menu relevance); re-ask or escalate when low
  - Memory trimming/dedup of summary for long sessions
- Planning
  - Add a planner (intent classifier) ahead of RAGService to route: Quote vs Info vs Escalation vs Logistics
  - Introduce a short “think” step before answering when low-confidence
- Observability
  - Structured event logs for: user msg, intent, retrieved IDs, answer type, quote state, totals, errors
  - Traces for feature flags and prompt version (PROMPT_VERSION)
- Evaluation
  - Golden-path tests for: menu relevance, numeric selection, PDF flow, price guard, sample-collection guidance, address reply
  - Regression assertions on answer length/tone and lack of invented prices
- Roadmap
  - Multi-quantity and per-test sample counts in quotes
  - Discount/tax handling (with visible line items)
  - Admin console for catalog edits and re-index
  - Multi-modal (image of sample container, PDF parsing of prior results)
  - Multi-tenant config (branding, address, disclaimers)

Repository map
- api/server.py — FastAPI routes, session management, quote endpoints
- app/rag_chain.py — RAG + quote flow + streaming
- app/retriever.py — FAISS load and retriever
- app/memory.py — version-safe ConversationSummaryBufferMemory wrapper (+SQLite)
- app/state.py — ChatSession and SelectedTest (stateful quoting)
- app/prepare_embeddings.py — normalize/clean catalog; write compiled JSON/CSV
- app/ingest.py — build FAISS vector store from compiled data
- app/prompts.py — system/user prompts, style guide, principle loaders
- app/pdf.py — formal quote PDF generator
- web/ — minimal chat UI and saved quotes
- data/compiled/ — compiled catalog (source of truth for ingest)
- tests/ — basic integration tests (retrieval, ingest, RAG)

Advanced trends (2024–2025) — what’s included now
- Sentiment-aware responses: detects strong negative sentiment and adjusts tone; offers escalation. (ENABLE_SENTIMENT=1)
- Proactive assistance: greeting/help triggers show quick actions (quote, sample steps, drop-off, consultant). (ENABLE_PROACTIVE_SUGGESTIONS=1)
- Agentic refine (lightweight): one extra retrieval pass with a topic hint before fallback. (ENABLE_AGENTIC_REFINE=1)
- Long-term continuity: rolling conversation summary memory with optional SQLite persistence.
- Guardrails: price integrity (catalog-only), location-only answers, selection-state quoting.

Roadmap
- Full agentic RAG planner (LangGraph) to decompose tasks and call tools (ERP pricing, CRM, email).
- Multimodal RAG: index PDFs/images of lab guides; support image Q&A.
- Personalisation: opt-in user profiles for preferred tests/channels; proactive reminders (POPIA-compliant).
- Evaluation: multi-turn conversation metrics; dashboards and drift monitoring.
- Voice/WhatsApp parity: add ASR/TTS for hands-free use; harden Twilio webhook flows.

Toggles (env)
- ENABLE_SENTIMENT, ENABLE_PROACTIVE_SUGGESTIONS, ENABLE_AGENTIC_REFINE, RETURN_DEBUG_TRACE
