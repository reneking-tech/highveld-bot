"""Conversation summary memory.

Keeps a compact rolling summary so follow-ups have context without bloating prompts.
Facts from retrieved context should remain authoritative over memory.
"""

from __future__ import annotations

from typing import Optional, Dict
import sqlite3
import os

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

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
        cur.execute("SELECT summary FROM conversation_memory WHERE session_id = ?", (session_id,))
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


def build_summary_memory(
    *,
    llm: Optional[ChatOpenAI] = None,
    max_token_limit: Optional[int] = None,
) -> ConversationSummaryBufferMemory:
    """Factory for a summary memory instance.

    - Uses ChatOpenAI with low temperature so summaries are consistent.
    - max_token_limit controls how large the rolling summary can grow
      before older details are compressed.
    """
    chat = llm or ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    limit = max_token_limit if max_token_limit is not None else MEMORY_TOKEN_LIMIT
    return ConversationSummaryBufferMemory(llm=chat, max_token_limit=limit)


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
        self._memory = build_summary_memory(llm=llm, max_token_limit=max_token_limit)
        self._session_id = session_id or "default"
        self._persisted_summary = ""
        if ENABLE_PERSISTENT_MEMORY:
            _init_db(MEMORY_DB_PATH)
            self._persisted_summary = _db_load_summary(MEMORY_DB_PATH, self._session_id)

    # --- Core API --- #
    def save_turn(self, user_message: str, ai_message: str) -> None:
        """Record a single conversational turn into memory."""
        self._memory.save_context({"input": user_message}, {"output": ai_message})
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

    def reset(self) -> None:
        """Clear all stored memory (new conversation)."""
        self._memory.clear()
        self._persisted_summary = ""
        if ENABLE_PERSISTENT_MEMORY:
            _db_save_summary(MEMORY_DB_PATH, self._session_id, "")

    # --- Advanced escape hatch --- #
    @property
    def raw(self) -> ConversationSummaryBufferMemory:
        """Access the underlying memory object if you need to wire it directly
        into a LangChain ConversationChain, etc.
        """
        return self._memory