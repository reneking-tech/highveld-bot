Here’s a solid README.md scaffold for highveld_bot/ 
Highveld Bot

An AI assistant for Highveld Biotech that answers client questions about lab testing, pricing, and logistics.
Built with LangChain to demonstrate how language models + retrieval can provide reliable, source-grounded answers.


 Project Goals
- Provide concise, professional answers to customer questions.
-  Always cite sources (so clients can trust the response).
- Escalate complex queries to human follow-up (e.g., book a call with business development team.
-  Appear helpful and approachable, not overly technical.


📂 Repository Structure

highveld_bot/
  README.md
  requirements.txt
  .env

  data/
    lab_faq.csv            # core knowledge base
    other_sources/

  notebooks/
    00_tracer_bullet.ipynb # quick experiments

  app/
    ingest.py              # data loaders, embeddings, vector store
    retriever.py           # retrieval logic
    prompts.py             # system & user prompt templates
    rag_chain.py           # builds retrieval-augmented chain
    memory.py              # conversation memory (optional, later)
    router.py              # escalation to human/book-a-call (later)

  eval/
    eval_set.jsonl         # {question, expected_answer}
    eval_runner.py         # test harness for accuracy

  cli.py                   # simple terminal chat
  streamlit_app.py         # later: demo UI


Build Milestones

Phase 1 — Tracer Bullet MVP
	•	Ingest lab_faq.csv → embed → vector store.
	•	Retrieval-augmented chain with {context} + {question}.
	•	CLI app to ask one question and get answer + cited source.
	•	Create eval_set.jsonl (10 customer Qs).

Phase 2 — Improve Quality
	•	Tune chunk size & retriever top_k.
	•	Prompt refinement: require exact quotes for prices/dates.
	•	Evaluate accuracy (target ≥80% contains-check).

Phase 3 — Memory & UX
	•	Add conversation memory for follow-up queries.
	•	Add escalation route for “complex query → book a call”.
	•	Lightweight Streamlit demo with chat interface.

Phase 4 — Extra Tools (Optional)
	•	Calculator for turnaround times.
	•	Quote generator (draft emails).
	•	Multi-source ingestion (PDFs, spreadsheets, website data).

⸻

🔍 Evaluation
	•	Use eval_runner.py to test bot on eval_set.jsonl.
	•	Track:
	•	Accuracy (% correct or partially correct answers).
	•	Faithfulness (no hallucinations).
	•	Latency (response time).


New features and settings
- Price integrity guard (no invented prices).
- Score/distance gate for weak retrieval matches.
- Prompt versioning for traceability.
- Optional persistent memory (SQLite) for conversation summaries.
- Simple per-IP rate limiting.
- SSE streaming endpoint.

Environment variables
- PROMPT_VERSION=2025-08-25-1
- SIMILARITY_DISTANCE_MAX=0.6
- ENABLE_PRICE_GUARD=1
- ENABLE_PERSISTENT_MEMORY=1
- RATE_LIMIT_PER_MIN=60

Streaming endpoints
- POST /stream (text/plain)
- POST /stream/sse (text/event-stream)
