"""Conversation summary memory.

Keeps a compact rolling summary so follow-ups have context without bloating prompts.
Facts from retrieved context should remain authoritative over memory.
"""

from __future__ import annotations
from langchain_openai import ChatOpenAI

from typing import Optional, Dict, Any, List, Tuple
import sqlite3
import os
import warnings
from inspect import signature
import json

from app.state import ChatSession, SelectedTest

# Suppress LangChain deprecation warning for ConversationSummaryBufferMemory
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="langchain.memory")
warnings.filterwarnings(
    "ignore",
    message="Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/",
    module="langchain.memory",
)
warnings.filterwarnings(
    "ignore", message=".*ConversationSummaryBufferMemory.*", module="langchain.memory"
)

# --- version-tolerant imports for LangChain (v0.1.x vs v0.2+) ---
try:
    # new split in newer versions
    from langchain_community.memory import ConversationSummaryBufferMemory
except Exception:
    try:
        from langchain.memory import ConversationSummaryBufferMemory  # type: ignore
    except Exception:
        ConversationSummaryBufferMemory = None  # type: ignore

# --- NEW: No-op fallback memory (runs without OpenAI/LangChain) --- #


class _NoopSummaryMemory:
    """Minimal drop-in replacement exposing save_context/load_memory_variables/clear."""

    def __init__(self, *_, **__):
        self._history: list[tuple[str, str]] = []

    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        self._history.append(
            (str(inputs.get("input", "")), str(outputs.get("output", ""))))
        # keep last ~10 turns compact
        if len(self._history) > 10:
            self._history = self._history[-10:]

    def load_memory_variables(self, _: Dict[str, Any]) -> Dict[str, str]:
        # Join into a compact pseudo-summary string
        parts: list[str] = []
        for u, a in self._history[-6:]:
            if u:
                parts.append(f"User: {u}")
            if a:
                parts.append(f"AI: {a}")
        return {"history": "\n".join(parts)}

    def clear(self) -> None:
        self._history.clear()


def _openai_kwargs() -> dict:
    kw: dict = {}
    base = (
        os.getenv("OPENAI_BASE_URL", "")
        or os.getenv("OPENAI_API_BASE", "")
        or "https://api.openai.com/v1"
    ).strip()
    if not base.startswith(("http://", "https://")):
        base = f"https://{base}"
    kw["base_url"] = base
    return kw


# Optional knobs from config (fallbacks provided if not defined)
try:
    from app.config import LLM_MODEL  # type: ignore
except Exception:
    LLM_MODEL = "gpt-4o-mini"

try:
    from app.config import LLM_TEMPERATURE  # type: ignore
except Exception:
    LLM_TEMPERATURE = 0.1

try:
    from app.config import MEMORY_TOKEN_LIMIT  # type: ignore
except Exception:
    # Rough budget for a compact rolling summary; tune as needed
    MEMORY_TOKEN_LIMIT = 600

try:
    from app.config import ENABLE_PERSISTENT_MEMORY, MEMORY_DB_PATH  # type: ignore
except Exception:
    ENABLE_PERSISTENT_MEMORY = False
    MEMORY_DB_PATH = os.path.join(os.getcwd(), ".data", "memory.sqlite3")


def _init_db(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_memory (
                session_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL DEFAULT '',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        con.commit()
    finally:
        con.close()


def _db_load_summary(path: str, session_id: str) -> str:
    try:
        con = sqlite3.connect(path)
        cur = con.cursor()
        cur.execute(
            "SELECT summary FROM conversation_memory WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        return row[0] if row and row[0] else ""
    except Exception:
        return ""
    finally:
        try:
            con.close()
        except Exception:
            pass


def _db_save_summary(path: str, session_id: str, summary: str) -> None:
    try:
        con = sqlite3.connect(path)
        cur = con.cursor()
        cur.execute(
            "INSERT INTO conversation_memory(session_id, summary) VALUES(?, ?)\n"
            "ON CONFLICT(session_id) DO UPDATE SET summary=excluded.summary, updated_at=CURRENT_TIMESTAMP",
            (session_id, summary),
        )
        con.commit()
    except Exception:
        pass
    finally:
        try:
            con.close()
        except Exception:
            pass


def _get_summary_memory(llm, max_token_limit: int = 1200, return_messages: bool = True, memory_key: str = "history", summary_llm=None):
    """
    Return a ConversationSummaryBufferMemory across LC versions.
    Falls back to ConversationBufferMemory if summary memory is unavailable.
    """
    if ConversationSummaryBufferMemory is None:
        try:
            from langchain.memory import ConversationBufferMemory  # type: ignore
        except Exception:
            # Last-resort no-op shim with same interface
            class _NoOpMemory:  # minimal compat
                def __init__(self, *_, **__): ...
                def load_memory_variables(self, *_, **__): return {}
                def save_context(self, *_, **__): ...
            return _NoOpMemory()
        return ConversationBufferMemory(return_messages=return_messages)

    try:
        if summary_llm:
            return _get_summary_memory(
                summary_llm=summary_llm,
                return_messages=return_messages,
                max_token_limit=max_token_limit,
                memory_key=memory_key,
            )
        else:
            return _get_summary_memory(
                llm=llm,
                return_messages=return_messages,
                max_token_limit=max_token_limit,
                memory_key=memory_key,
            )
    except TypeError:
        # older signature compatibility
        if summary_llm:
            return _get_summary_memory(
                summary_llm=summary_llm,
                return_messages=return_messages,
                max_token_limit=max_token_limit,
                memory_key=memory_key,
            )
        else:
            return _get_summary_memory(
                llm=llm,
                return_messages=return_messages,
                max_token_limit=max_token_limit,
                memory_key=memory_key,
            )


def build_summary_memory(
    *,
    llm: Optional[ChatOpenAI] = None,
    max_token_limit: Optional[int] = None,
) -> Any:
    """Factory for a summary memory instance.

    - Uses ChatOpenAI with low temperature so summaries are consistent.
    - max_token_limit controls how large the rolling summary can grow.
    - Adapts to LangChain versions expecting either llm= or summary_llm=.
    - Falls back to a local no-op memory if LC/OpenAI is unavailable.
    """
    # Fallback immediately if LC memory class is missing
    if ConversationSummaryBufferMemory is None:  # type: ignore
        return _NoopSummaryMemory()

    # Try to build an OpenAI client; if this fails (e.g., no API key), use noop
    try:
        chat = llm or ChatOpenAI(
            model=LLM_MODEL, temperature=LLM_TEMPERATURE, **_openai_kwargs())
    except Exception:
        return _NoopSummaryMemory()

    limit = max_token_limit if max_token_limit is not None else MEMORY_TOKEN_LIMIT

    # Prefer probing the constructor signature; fallback on any exception
    try:
        # type: ignore[arg-type]
        params = set(
            signature(ConversationSummaryBufferMemory).parameters.keys())
    except Exception:
        params = {"llm"}  # safest default for older versions

    try:
        if "summary_llm" in params:
            return _get_summary_memory(None, limit, return_messages=False, memory_key="history", summary_llm=chat)
        # Older LC versions expect llm=
        return _get_summary_memory(chat, limit, return_messages=False, memory_key="history")
    except Exception:
        # Last-resort: local noop
        return _NoopSummaryMemory()


class MemoryService:
    """Wrapper around ConversationSummaryBufferMemory.

    Provides explicit methods to save a turn, load the current summary,
    and reset memory. Keeps rag_chain.py clean.
    """

    def __init__(
        self,
        *,
        llm: Optional[ChatOpenAI] = None,
        max_token_limit: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self._memory = build_summary_memory(
            llm=llm, max_token_limit=max_token_limit)
        self._session_id = session_id or "default"
        self._persisted_summary = ""
        self._recent_turns = []  # Store recent conversation turns
        self._max_recent_turns = 5  # Keep last 5 turns for immediate context
        if ENABLE_PERSISTENT_MEMORY:
            _init_db(MEMORY_DB_PATH)
            self._persisted_summary = _db_load_summary(
                MEMORY_DB_PATH, self._session_id)

    # --- Core API --- #
    def save_turn(self, user_message: str, ai_message: str) -> None:
        """Record a single conversational turn into memory."""
        self._memory.save_context({"input": user_message}, {
                                  "output": ai_message})

        # Add to recent turns
        self._recent_turns.append((user_message, ai_message))
        # Keep only the most recent turns
        if len(self._recent_turns) > self._max_recent_turns:
            self._recent_turns = self._recent_turns[-self._max_recent_turns:]

        # Merge new summary with any persisted one, then persist
        try:
            vars: Dict[str, str] = self._memory.load_memory_variables({})
            current = vars.get("history", "")
        except Exception:
            current = ""
        if self._persisted_summary:
            # Simple concatenation; could be improved with dedupe.
            merged = (self._persisted_summary + "\n" + current).strip()
        else:
            merged = current
        self._persisted_summary = merged
        if ENABLE_PERSISTENT_MEMORY:
            _db_save_summary(MEMORY_DB_PATH, self._session_id, merged)

    def load_summary(self) -> str:
        """Return the current rolling summary (empty string if none)."""
        if self._persisted_summary:
            return self._persisted_summary
        vars: Dict[str, str] = self._memory.load_memory_variables({})
        # ConversationSummaryBufferMemory stores the running summary in "history"
        return vars.get("history", "")

    def get_recent_turns(self, count: int = 3) -> List[Tuple[str, str]]:
        """Get the most recent conversation turns (user, ai) pairs."""
        return self._recent_turns[-count:] if count > 0 else []

    def get_formatted_history(self, max_turns: int = 3) -> str:
        """Format recent conversation history as a string."""
        recent = self.get_recent_turns(max_turns)
        lines = []
        for user_msg, ai_msg in recent:
            lines.append(f"User: {user_msg}")
            lines.append(f"Assistant: {ai_msg}")
            lines.append("")
        return "\n".join(lines).strip()

    def reset(self) -> None:
        """Clear all stored memory (new conversation)."""
        self._memory.clear()
        self._persisted_summary = ""
        self._recent_turns = []
        if ENABLE_PERSISTENT_MEMORY:
            _db_save_summary(MEMORY_DB_PATH, self._session_id, "")

    def save_session(self, session: ChatSession) -> None:
        """Save ChatSession state for persistence across API calls."""
        data = {
            "selected_tests": [
                {"code": t.code, "name": t.name,
                    "price": t.price, "tat_days": t.tat_days}
                for t in session.selected_tests
            ],
            "last_options": [
                {"code": t.code, "name": t.name,
                    "price": t.price, "tat_days": t.tat_days}
                for t in session.last_options
            ],
            "state": session.state,
            "last_user_free_text": session.last_user_free_text,
            "pending_quote_id": session.pending_quote_id,
            "escalate_field": session.escalate_field,
            "escalate": session.escalate,
            "mode": session.mode,
            "client_name": getattr(session, "client_name", ""),
            "client_surname": getattr(session, "client_surname", ""),
            "client_company": getattr(session, "client_company", ""),
            "client_email": getattr(session, "client_email", ""),
            "client_extra_info": getattr(session, "client_extra_info", ""),
            "client_phone": getattr(session, "client_phone", ""),
            "sample_counts": getattr(session, "sample_counts", {}),
            "collect_counts_index": getattr(session, "collect_counts_index", 0),
        }
        self._persisted_session = data
        if ENABLE_PERSISTENT_MEMORY:
            _db_save_summary(MEMORY_DB_PATH, self._session_id +
                             "_session", json.dumps(data))

    def load_session(self) -> Optional[ChatSession]:
        """Load ChatSession state if available."""
        data = None
        if hasattr(self, "_persisted_session") and self._persisted_session:
            data = self._persisted_session
        elif ENABLE_PERSISTENT_MEMORY:
            summary = _db_load_summary(
                MEMORY_DB_PATH, self._session_id + "_session")
            if summary:
                try:
                    data = json.loads(summary)
                except Exception:
                    return None
        if not data:
            return None
        session = ChatSession()
        session.selected_tests = [
            SelectedTest(code=t["code"], name=t["name"],
                         price=t["price"], tat_days=t["tat_days"])
            for t in data.get("selected_tests", [])
        ]
        session.last_options = [
            SelectedTest(code=t["code"], name=t["name"],
                         price=t["price"], tat_days=t["tat_days"])
            for t in data.get("last_options", [])
        ]
        session.state = data.get("state", "idle")
        session.last_user_free_text = data.get("last_user_free_text", "")
        session.pending_quote_id = data.get("pending_quote_id", None)
        session.escalate_field = data.get("escalate_field", None)
        session.escalate = data.get("escalate", {})
        session.mode = data.get("mode", None)
        session.client_name = data.get("client_name", "")
        session.client_surname = data.get("client_surname", "")
        session.client_company = data.get("client_company", "")
        session.client_email = data.get("client_email", "")
        session.client_extra_info = data.get("client_extra_info", "")
        session.client_phone = data.get("client_phone", "")
        session.sample_counts = data.get("sample_counts", {}) or {}
        session.collect_counts_index = int(
            data.get("collect_counts_index", 0) or 0)
        return session

    # --- Advanced escape hatch --- #
    @property
    def raw(self) -> "ConversationSummaryBufferMemory":  # type: ignore
        """Access the underlying memory object if you need to wire it directly
        into a LangChain ConversationChain, etc.
        """
        return self._memory
        return self._memory
