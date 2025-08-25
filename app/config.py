"""Central configuration for the RAG service.

Reads environment variables with sensible defaults for local development.
Keep this file small and declarative; add only simple constants and flags here.
"""

from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables once for the whole app
load_dotenv(find_dotenv())

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # repo root
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "lab_faq.csv")

# Persisted vector index
INDEX_DIR = os.path.join(PROJECT_ROOT, ".vectordb", "faq_faiss")

# Ingestion knobs
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Global knobs
TOP_K = 4
EMBEDDING_MODEL = "text-embedding-3-large"

# Optional LLM config (used by services/UI)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "2025-08-25-1")

# Canonical physical drop-off address (used for location queries)
DROP_OFF_ADDRESS = (
    "Physical Address: Modderfontein Industrial Complex, Standerton Avenue, via Nobel Gate, "
    "Modderfontein, Gauteng, South Africa, 1645. Samples can be dropped at the permit office "
    "on the corner of Nobel Avenue and Standerton Road."
)

# Confidence/uncertainty heuristics
# Minimum fraction of unique keywords from the question that should appear in the
# retrieved context before we trust an LLM answer. If below this threshold and
# ENABLE_NOT_SURE_FALLBACK is true, the bot will say it's not sure and ask to clarify.
MIN_KEYWORD_OVERLAP = float(os.getenv("MIN_KEYWORD_OVERLAP", "0.12"))
ENABLE_NOT_SURE_FALLBACK = os.getenv(
    "ENABLE_NOT_SURE_FALLBACK", "1").lower() not in {"0", "false", "no"}

# Optional score/distance-based gate using the vectorstore's top match distance.
# If the best match distance exceeds this max, we prefer to say "not sure".
# Note: For FAISS cosine distance, smaller is better; tune as needed.
SIMILARITY_DISTANCE_MAX = float(os.getenv("SIMILARITY_DISTANCE_MAX", "0.6"))

# Business guardrails
# Price integrity guard: avoid invented prices
ENABLE_PRICE_GUARD = os.getenv("ENABLE_PRICE_GUARD", "1").lower() not in {"0", "false", "no"}

# Memory (SQLite) — summaries persisted per session
DATA_DIR = os.path.join(PROJECT_ROOT, ".data")
os.makedirs(DATA_DIR, exist_ok=True)
MEMORY_DB_PATH = os.path.join(DATA_DIR, "memory.sqlite3")

# Toggle persistent memory on/off
ENABLE_PERSISTENT_MEMORY = os.getenv("ENABLE_PERSISTENT_MEMORY", "1").lower() not in {"0", "false", "no"}

# Simple API rate limiting (per-IP, per minute)
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
