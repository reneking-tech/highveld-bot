"""Central configuration for the RAG service.

Adds a typed Settings object (Pydantic BaseSettings) for dependency injection
in the API. Existing module-level constants remain for backward compatibility
but should be gradually migrated to `Settings`.
"""

from dotenv import load_dotenv, find_dotenv
import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except Exception:  # pragma: no cover
    from pydantic import BaseSettings  # type: ignore

# Load environment variables once for the whole app
load_dotenv(find_dotenv())

# Normalise OPENAI_BASE_URL so downstream clients don't see an invalid URL


def _sanitize_openai_base() -> None:
    base = os.getenv("OPENAI_BASE_URL", "").strip()
    api_base = os.getenv("OPENAI_API_BASE", "").strip()
    use = base or api_base
    if not use:
        # Remove empty vars to let SDK default to https://api.openai.com/v1
        os.environ.pop("OPENAI_BASE_URL", None)
        os.environ.pop("OPENAI_API_BASE", None)
        return
    if not (use.startswith("http://") or use.startswith("https://")):
        use = "https://" + use
    # Set both to the same sanitised value for compatibility
    os.environ["OPENAI_BASE_URL"] = use
    os.environ["OPENAI_API_BASE"] = use


_sanitize_openai_base()


class Settings(BaseSettings):
    """Runtime settings for the API and services.

    Values are loaded from environment variables and optional .env files.
    """

    APP_NAME: str = "Highveld Biotech Chatbot"
    ENV: str = os.getenv("ENV", "dev")

    # Feature flags
    STREAMING: bool = False
    TRACE_LOGGING: bool = False
    STRICT_MODE: bool = False

    # Retrieval/indexing
    VECTOR_INDEX_PATH: str = os.getenv(
        "VECTOR_INDEX_PATH",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".vectordb", "faq_faiss"),
    )
    EMBED_MODEL: str = os.getenv(
        "EMBED_MODEL", os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    )
    MAX_CHUNKS: int = int(os.getenv("MAX_CHUNKS", "6"))
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    FETCH_K: int = int(os.getenv("FETCH_K", "20"))

    # PDF / quoting
    PDF_FONT_PATH: Optional[str] = os.getenv("PDF_FONT_PATH") or None
    QUOTE_VALIDITY_DAYS: int = int(os.getenv("QUOTE_VALIDITY_DAYS", "14"))

    class Config:
        env_file = ".env"
        case_sensitive = False


# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # repo root
# DATA_FILE (prefer the simple lab_faq CSV during development; fall back to compiled)
_default_data = os.path.join(PROJECT_ROOT, "data", "lab_faq.csv")
_compiled_fallback = os.path.join(PROJECT_ROOT, "data", "compiled", "lab_tests_clean.csv")
if os.path.exists(_default_data):
    DATA_FILE = _default_data
else:
    DATA_FILE = _compiled_fallback

# Persisted vector index (legacy constant)
INDEX_DIR = os.path.join(PROJECT_ROOT, ".vectordb", "faq_faiss")

# Ingestion knobs
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Global knobs
TOP_K = int(os.getenv("RAG_TOP_K", "6"))  # Increased for detailed lab queries
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# Optional LLM config (used by services/UI)
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "2025-08-25-1")
RETURN_DEBUG_TRACE = os.getenv("RETURN_DEBUG_TRACE", "0").lower() not in {"0", "false", "no"}

# Canonical physical drop-off address (used for location queries)
DROP_OFF_ADDRESS = (
    "Physical Address: Modderfontein Industrial Complex, Standerton Avenue, via Nobel Gate, "
    "Modderfontein, Gauteng, South Africa, 1645. Samples can be dropped at the permit office "
    "on the corner of Nobel Avenue and Standerton Road."
)

# Lab hours for drop-off (displayed in guidance)
LAB_HOURS = "Sample reception hours: Mon–Thu 08:00–16:00, Fri 08:00–14:00."

# Quotation and payment policy
QUOTE_VALIDITY_DAYS = int(os.getenv("QUOTE_VALIDITY_DAYS", "14"))
PAYMENT_POLICY = "Results are released after payment is received. Please email proof of payment to renesking@gmail.com."

# Banking details (used on quote/invoice notes)
BANK_DETAILS = (
    "Banking details (FNB Savings):\n"
    "Name: Rene S King\n"
    "Account: 62369338407\n"
    "Branch code: 250655\n"
    "SWIFT: FIRNZAJJ"
)

# Rush fee policy
RUSH_FEE_THRESHOLD_ZAR = float(os.getenv("RUSH_FEE_THRESHOLD_ZAR", "10000"))
RUSH_FEE_RATE = float(os.getenv("RUSH_FEE_RATE", "0.30"))

# Confidence/uncertainty heuristics
# Minimum fraction of unique keywords from the question that should appear in the
# retrieved context before we trust an LLM answer. If below this threshold and
# ENABLE_NOT_SURE_FALLBACK is true, the bot will say it's not sure and ask to clarify.
MIN_KEYWORD_OVERLAP = float(os.getenv("MIN_KEYWORD_OVERLAP", "0.05"))
ENABLE_NOT_SURE_FALLBACK = os.getenv("ENABLE_NOT_SURE_FALLBACK", "1").lower() not in {
    "0",
    "false",
    "no",
}

# Optional score/distance-based gate using the vectorstore's top match distance.
# If the best match distance exceeds this max, we prefer to say "not sure".
# Note: For FAISS cosine distance, smaller is better; tune as needed.
SIMILARITY_DISTANCE_MAX = float(os.getenv("SIMILARITY_DISTANCE_MAX", "0.6"))

# Business guardrails
# Price integrity guard: avoid invented prices
ENABLE_PRICE_GUARD = os.getenv("ENABLE_PRICE_GUARD", "1").lower() not in {"0", "false", "no"}

# --- Advanced trends feature flags (lightweight implementations) --- #
ENABLE_SENTIMENT = os.getenv("ENABLE_SENTIMENT", "1").lower() not in {"0", "false", "no"}
ENABLE_PROACTIVE_SUGGESTIONS = os.getenv("ENABLE_PROACTIVE_SUGGESTIONS", "1").lower() not in {
    "0",
    "false",
    "no",
}
ENABLE_AGENTIC_REFINE = os.getenv("ENABLE_AGENTIC_REFINE", "1").lower() not in {"0", "false", "no"}

# Memory (SQLite) — summaries persisted per session
DATA_DIR = os.path.join(PROJECT_ROOT, ".data")
os.makedirs(DATA_DIR, exist_ok=True)
MEMORY_DB_PATH = os.path.join(DATA_DIR, "memory.sqlite3")

# Toggle persistent memory on/off
ENABLE_PERSISTENT_MEMORY = os.getenv("ENABLE_PERSISTENT_MEMORY", "1").lower() not in {
    "0",
    "false",
    "no",
}

# Simple API rate limiting (per-IP, per minute)
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))

# Conversation enhancement settings
ENABLE_QUESTION_UNDERSTANDING = os.getenv("ENABLE_QUESTION_UNDERSTANDING", "1").lower() not in {
    "0",
    "false",
    "no",
}
ENABLE_ANSWER_REFINEMENT = os.getenv("ENABLE_ANSWER_REFINEMENT", "1").lower() not in {
    "0",
    "false",
    "no",
}
# Max follow-up questions before proceeding
MAX_CLARIFICATION_ROUNDS = int(os.getenv("MAX_CLARIFICATION_ROUNDS", "2"))

# NEW: optional LLM-assisted intent classification (off by default for reliability)
ENABLE_LLM_INTENT_CLASSIFIER = os.getenv("ENABLE_LLM_INTENT_CLASSIFIER", "1").lower() not in {
    "0",
    "false",
    "no",
}

# Enhanced retrieval settings
ENABLE_QUERY_REWRITING = os.getenv("ENABLE_QUERY_REWRITING", "1").lower() not in {
    "0",
    "false",
    "no",
}
# Weight for entity-based retrievals
ENTITY_BOOST_WEIGHT = float(os.getenv("ENTITY_BOOST_WEIGHT", "0.3"))

# Enable technical handling for lab-specific queries
ENABLE_TECHNICAL_HANDLING = os.getenv("ENABLE_TECHNICAL_HANDLING", "1").lower() not in {
    "0",
    "false",
    "no",
}
