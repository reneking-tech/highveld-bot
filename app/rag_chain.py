"""
RAG chains for the Highveld Biotech chatbot.

Provides two public entry points:
- RAGService.answer_question(question: str) -> dict   # prose answer + sources
- RAGService.structured_quote(question: str) -> dict
"""

from __future__ import annotations

import json
import math
import re
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Optional LangChain imports with safe fallbacks
try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:  # pragma: no cover

    class ChatOpenAI:  # type: ignore
        def __init__(self, *_, **__):
            pass

        def invoke(self, *_args, **_kwargs):
            class _R:
                content = ""

            return _R()

        def stream(self, *_args, **_kwargs):
            class _S:
                content = ""

            yield _S()


try:
    from langchain.schema import SystemMessage, HumanMessage  # type: ignore
except Exception:  # pragma: no cover

    class _Msg:
        def __init__(self, content: str):
            self.content = content

    SystemMessage = _Msg  # type: ignore
    HumanMessage = _Msg  # type: ignore

import app.prompts as prompts
import app.config as cfg
from app.retriever import get_retriever, top_similarity_distance
from app.state import ChatSession, SelectedTest
from app.quote_flow import handle_quote_flow

# For static type checkers (avoid runtime import cycles)
if TYPE_CHECKING:  # pragma: no cover
    from app.memory import MemoryService  # noqa: F401

# ---- Import helpers from intent_helpers (single source of truth) ----
from app.intent_helpers import (
    _is_greeting,
    _is_acknowledgement,
    _is_escalation_request,
    _is_sample_collection_query,
    _is_location_query,
    _tokenize,
    _get_test_names,
    _parse_doc_content,
    _retrieve_docs,
    _maybe_extract_answer_field,
    _extract_recent_topic,
    _question_sentence_from,
    _sentiment_score,
    _quick_suggestions,
    _ensure_json,
    _post_validate_quote,
    _coerce_number,
    _docs_to_options,
    _is_price_query,
    _is_tat_query,
)

# Optional kwargs dict (be resilient if it doesn’t exist in intent_helpers)
try:
    from app.intent_helpers import _openai_kwargs as _OPENAI_KWARGS  # type: ignore
except Exception:
    _OPENAI_KWARGS: Dict[str, Any] = {}

logger = logging.getLogger(__name__)


# ---------------- Utility helpers (no regex re-definitions here) ----------------


def _format_context(docs: List[Any]) -> Tuple[str, List[str]]:
    """Compact context text and list of IDs/snippets for the prompt."""
    lines: List[str] = []
    ids: List[str] = []
    for doc in docs or []:
        ids.append(_extract_id(doc))
        lines.append(_short_snippet(doc))
    return "\n".join(lines), ids


def _extract_id(doc) -> str:
    """Best-effort extraction of the HVB ID from the document content."""
    try:
        d = _parse_doc_content(getattr(doc, "page_content", "") or "")
        if d.get("id"):
            return str(d["id"]).strip()
        m = re.search(r"\bHVB-\d{4}\b", getattr(doc, "page_content", "") or "")
        if m:
            return m.group(0)
        row = (getattr(doc, "metadata", {}) or {}).get("row")
        if row is not None:
            return f"ROW-{row}"
    except Exception:
        pass
    return "UNKNOWN-ID"


def _short_snippet(doc) -> str:
    """Build a short human-friendly snippet using key fields if available."""
    d = {}
    try:
        d = _parse_doc_content(getattr(doc, "page_content", "") or "")
    except Exception:
        d = {}

    name = d.get("test_name") or d.get("category") or ""
    price = d.get("price_zar") or d.get("price_ZAR") or d.get("price") or ""
    tat = d.get("turnaround_days") or d.get("tat") or ""
    addr = d.get("address", "")
    category = (d.get("category") or "").lower()

    bits: List[str] = []
    if category in ("drop_off", "drop-off", "address"):
        title = name or "Sample drop-off"
        bits.append(title)
        if addr:
            bits.append(addr)
        if tat:
            bits.append(f"TAT {tat} days")
    else:
        if name:
            bits.append(name)
        if price:
            p_str = str(price).replace(",", "").replace("R", "").strip()
            try:
                p = float(p_str)
                if p > 0:
                    bits.append(f"R{p:,.2f}")
            except Exception:
                # If not parseable, just show the raw price text
                bits.append(str(price))
        if tat:
            bits.append(f"TAT {tat} days")

    return " — ".join([b for b in bits if b])[:220]


def _tokenize_safe(text: str) -> List[str]:
    return _tokenize(text)


def _keyword_overlap_ratio(question: str, context_text: str) -> float:
    q = set(_tokenize_safe(question))
    c = set(_tokenize_safe(context_text))
    if not q or not c:
        return 0.0
    return len(q & c) / max(1, len(q))


def _contains_price(text: str) -> bool:
    return bool(re.search(r"\bR?\s?\d[\d,]*(\.\d{2})?\b", text or ""))


def _summarize_selection(tests: list[SelectedTest], *, show_price: bool, show_tat: bool) -> str:
    lines: list[str] = ["Here is your current selection:"]
    total = 0.0
    for t in tests:
        parts = [t.name]
        if show_price:
            parts.append(f"R{t.price:,.2f}")
            total += float(t.price or 0.0)
        if show_tat and t.tat_days:
            parts.append(f"TAT {int(t.tat_days)} days")
        lines.append("- " + " • ".join(parts))
    if show_price:
        lines.append(f"Total: R{total:,.2f}")
    lines.append("Would you like me to proceed or adjust your selection?")
    return "\n".join(lines)


def _pick_model() -> tuple[str, float, int]:
    model = getattr(cfg, "LLM_MODEL", "gpt-4o-mini")
    temp = float(getattr(cfg, "LLM_TEMPERATURE", 0.2))
    mx = int(getattr(cfg, "MAX_TOKENS", 800))
    return model, temp, mx


AFFIRM = {
    "yes",
    "y",
    "ok",
    "okay",
    "sure",
    "confirm",
    "go ahead",
    "proceed",
    "quote",
    "pdf",
    "generate",
}
NEGATE = {"no", "nope", "nah", "that's all",
          "thats all", "done", "finish", "none"}


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def _fmt_option(i: int, t: SelectedTest) -> str:
    # Hide pricing/TAT by default (revealed on request)
    return f"{i}. {t.name}"


def _fmt_line_item(t: SelectedTest) -> str:
    # Hide pricing/TAT by default (revealed on request)
    return f"- {t.name}"


def _total(tests: list[SelectedTest]) -> float:
    return sum(t.price for t in tests)


def _match_user_choices(user_text: str, options: list[SelectedTest]) -> list[SelectedTest]:
    """Match user input (numbers or names) to available options. Chops off anything after '?'."""
    user_text = (user_text or "").lower().split("?")[0]
    chosen: list[SelectedTest] = []
    numbers = set(re.findall(r"\b\d+\b", user_text))
    for idx, opt in enumerate(options):
        if str(idx + 1) in numbers:
            chosen.append(opt)
    for opt in options:
        name = opt.name.lower()
        if name in user_text or any(word in user_text for word in name.split()):
            if opt not in chosen:
                chosen.append(opt)
    return chosen


def _looks_like_new_request(q: str) -> bool:
    if not q:
        return False
    s = (q or "").strip().lower()
    if _is_sample_collection_query(q):
        return True
    keywords = (
        "another",
        "also",
        "add",
        "more",
        "different",
        "new",
        "instead",
        "change",
        "switch",
        "soil",
        "water",
        "test",
        "tests",
    )
    return any(k in s for k in keywords)


def _coerce_tests(raw_docs) -> list[SelectedTest]:
    """Map retriever docs → SelectedTest list, skipping address/unpriced cards."""
    out: list[SelectedTest] = []
    for d in raw_docs or []:
        meta = getattr(d, "metadata", {}) or {}
        parsed = _parse_doc_content(getattr(d, "page_content", "") or "")

        cat = (str(meta.get("category") or parsed.get(
            "category") or "")).strip().lower()
        if cat in {"address", "drop_off", "drop-off"}:
            continue

        code = _extract_id(d)
        name = parsed.get("test_name") or meta.get(
            "test_name") or meta.get("name") or ""
        price = (
            meta.get("price_zar")
            or meta.get("price_ZAR")
            or parsed.get("price_ZAR")
            or parsed.get("price")
        )
        tat = (
            meta.get("tat_days")
            or meta.get("turnaround_days")
            or parsed.get("turnaround_days")
            or parsed.get("tat")
        )

        try:
            price_val = (
                float(str(price).replace("R", "").replace(",", "").strip())
                if price not in (None, "")
                else None
            )
        except Exception:
            price_val = None

        if (
            not name
            or price_val is None
            or (isinstance(price_val, (int, float)) and (math.isnan(price_val) or price_val <= 0))
        ):
            continue

        try:
            out.append(
                SelectedTest(
                    code=code or (name.lower()[:24] or "item"),
                    name=name,
                    price=price_val,
                    tat_days=int(float(tat)) if tat not in (None, "") else 0,
                )
            )
        except Exception:
            continue
    return out


# ------------------------------- Service -------------------------------


class RAGService:
    """High-level service that encapsulates retrieval + prompting."""

    def __init__(
        self,
        llm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        memory: Optional["MemoryService"] = None,
        session_id: Optional[str] = None,
    ) -> None:
        from app.memory import MemoryService  # local import to avoid cycles

        model_name, default_temp, default_max = _pick_model()
        model_name = llm_model or model_name
        temp = default_temp if temperature is None else float(temperature)
        k = int(top_k) if top_k is not None else int(getattr(cfg, "TOP_K", 4))

        self.llm = ChatOpenAI(
            model=model_name, temperature=temp, max_tokens=default_max, **_OPENAI_KWARGS
        )
        self.conversation_llm = ChatOpenAI(
            model=model_name,
            temperature=min(0.7, temp + 0.2),
            max_tokens=default_max,
            **_OPENAI_KWARGS,
        )

        self.retriever = get_retriever(top_k=k)
        self.mem = memory or MemoryService(session_id=session_id)
        self._config = {
            "model": model_name,
            "temperature": temp,
            "top_k": k,
            "max_tokens": default_max,
        }

        self._clarification_count = 0
        self._MAX_CLARIFICATIONS = 2

    def _rebuild_llm(self) -> None:
        self.llm = ChatOpenAI(
            model=self._config["model"],
            temperature=float(self._config["temperature"]),
            max_tokens=int(self._config.get("max_tokens", _pick_model()[2])),
            **_OPENAI_KWARGS,
        )

    def _maybe_adjust_generation(self, text: str) -> str | None:
        t = (text or "").lower()
        m = re.search(r"set (the )?temperature (to|at)\s*([0-9.]+)", t)
        if m:
            try:
                val = max(0.0, min(1.0, float(m.group(3))))
                self._config["temperature"] = val
                self._rebuild_llm()
                return f"Understood — I'll be a bit more conversational (temperature {val:.2f}). What would you like to test today?"
            except Exception:
                pass
        if re.search(
            r"(increase|raise|make.*higher).*(temperature|randomness)|be more (creative|conversational|engaging)",
            t,
        ):
            cur = float(self._config.get("temperature", 0.2))
            val = min(cur + 0.3, 0.9)
            self._config["temperature"] = val
            self._rebuild_llm()
            return f"Got it — I'll be more conversational (temperature {val:.2f}). Are you testing water or soil, or should I recommend options?"
        return None

    def _summary_message(self) -> List[Any]:
        summary = (self.mem.load_summary() or "").strip()
        return (
            [SystemMessage(
                content=f"Conversation summary (context, not facts):\n{summary}")]
            if summary
            else []
        )

    def _get_history_snippet(self, max_chars: int = 600) -> str:
        history_lines: List[str] = []
        try:
            recent = []
            if hasattr(self.mem, "get_recent"):
                recent = self.mem.get_recent(3)  # type: ignore
            elif hasattr(self.mem, "load_recent"):
                recent = self.mem.load_recent(3)  # type: ignore
            for item in recent or []:
                if isinstance(item, dict):
                    human = _maybe_extract_answer_field(
                        item.get("human", ""))  # type: ignore
                    ai = _maybe_extract_answer_field(
                        item.get("ai", ""))  # type: ignore
                elif isinstance(item, (tuple, list)) and len(item) >= 2:
                    human = _maybe_extract_answer_field(str(item[0]))
                    ai = _maybe_extract_answer_field(str(item[1]))
                else:
                    continue
                if human:
                    history_lines.append(f"User: {human}")
                if ai:
                    history_lines.append(f"AI: {ai}")
        except Exception:
            pass
        return "\n".join(history_lines)[-max_chars:]

    def _augment_follow_up(self, question: str) -> Tuple[str, str]:
        history = self._get_history_snippet()
        summary = ""
        try:
            summary = (self.mem.load_summary() or "").strip()
        except Exception:
            summary = ""
        if summary:
            q = (question or "").strip()
            return q, f"SUMMARY: {summary}\n\n{history}"
        q = (question or "").strip()
        if not _is_acknowledgement(q):
            return q, history
        topic = _extract_recent_topic(history)
        last_ai_question = _question_sentence_from(history)
        if last_ai_question:
            augmented = f"Please proceed with: {last_ai_question}"
            if topic:
                augmented += f" The recent topic was: {topic}."
            return augmented, history
        return q, history

    def _classify_intent_llm_safe(self, text: str, history: str = "") -> Dict[str, Any]:
        # Heuristic + (optional) your LLM classifier if enabled in cfg
        try:
            if getattr(cfg, "ENABLE_LLM_INTENT_CLASSIFIER", False):
                sys = SystemMessage(
                    content=(
                        "You are a compact intent classifier. Return strict JSON only with keys: "
                        '{"intent":"quote|info|logistics|escalation|other","certainty":0.0-1.0,"entities":[] }'
                    )
                )
                user = HumanMessage(
                    content=f"History:\n{(history or '')[-600:]}\n\nUser:\n{text}\n\nReturn JSON only."
                )
                resp = self.llm.invoke([sys, user])
                raw = resp.content if hasattr(resp, "content") else str(resp)
                m = re.search(r"(\{.*\})", raw, re.DOTALL)
                if m:
                    raw = m.group(1)
                obj = json.loads(raw)
                if isinstance(obj, dict) and obj.get("intent"):
                    return {"intent": obj.get("intent"), "entities": obj.get("entities", [])}
        except Exception:
            pass

        # Fallback heuristics
        lower_text = (text or "").lower()
        if _is_escalation_request(lower_text):
            return {"intent": "escalation", "entities": []}
        if any(k in lower_text for k in ("price", "cost", "quote", "pricing", "estimate")):
            return {"intent": "quote", "entities": []}
        if _is_location_query(lower_text) or any(k in lower_text for k in ("address", "drop-off", "where")):
            return {"intent": "logistics", "entities": []}
        test_names = _get_test_names()
        ents = [w for w in _tokenize_safe(text) if w in test_names]
        return {"intent": "info", "entities": ents}

    # ---------------------------- Public API ----------------------------

    def handle_quote_flow(self, user_text: str, session: ChatSession) -> str:
        return handle_quote_flow(user_text, session, self.retriever)

    def answer_question(
        self,
        question: str,
        *,
        extra_context: Optional[str] = None,
        session: Optional[ChatSession] = None,
    ) -> Dict[str, Any]:
        q = (question or "").strip()

        def _build_trace(**kwargs) -> Dict[str, Any]:
            tr = {
                "keyword_overlap": None,
                "top_similarity_distance": None,
                "intent": None,
                "analysis": None,
                "gated": None,
                "override": None,
            }
            tr.update({k: v for k, v in kwargs.items() if v is not None})
            return tr

        # Sentiment
        if getattr(cfg, "ENABLE_SENTIMENT", True) and _sentiment_score(q) <= -0.6:
            apology = "I apologize for any frustration. Let me help you more directly."
            try:
                self.mem.save_turn(question, apology)
            except Exception:
                pass
            return {
                "answer": apology
                + " What specifically are you looking for today? I can help with quotes, sample collection guidance, or connect you with a specialist.",
                "sources": [],
                "trace": _build_trace(intent="info"),
            }

        # Meta control (temperature)
        meta_msg = self._maybe_adjust_generation(q)
        if meta_msg:
            try:
                self.mem.save_turn(question, meta_msg)
            except Exception:
                pass
            return {"answer": meta_msg, "sources": [], "trace": _build_trace(intent="info")}

        # Early address/drop-off
        if _is_location_query(q):
            addr = getattr(cfg, "DROP_OFF_ADDRESS", "").strip()
            try:
                self.mem.save_turn(question, addr)
            except Exception:
                pass
            return {
                "answer": addr,
                "sources": [],
                "trace": _build_trace(intent="logistics", override="DROP_OFF_ADDRESS"),
            }

        # Greetings / quick suggestions
        if _is_greeting(q):
            text = (
                "Hello! I can help with laboratory testing — water, fertiliser/soil, or other chemical tests. "
                "Would you like a quote, sample preparation guidance, drop-off details, or to speak with a consultant?"
            )
            if getattr(cfg, "ENABLE_PROACTIVE_SUGGESTIONS", True):
                text += "\n\n" + _quick_suggestions()
            try:
                self.mem.save_turn(question, text)
            except Exception:
                pass
            return {"answer": text, "sources": [], "trace": _build_trace(intent="info")}

        if getattr(cfg, "ENABLE_PROACTIVE_SUGGESTIONS", True) and q.lower() in {
            "help",
            "options",
            "menu",
        }:
            txt = _quick_suggestions()
            try:
                self.mem.save_turn(question, txt)
            except Exception:
                pass
            return {"answer": txt, "sources": [], "trace": _build_trace(intent="info")}

        # Sample collection guidance
        if _is_sample_collection_query(q):
            addr = getattr(cfg, "DROP_OFF_ADDRESS", "").strip()
            hours = getattr(cfg, "LAB_HOURS", "").strip()
            guidance = (
                "Water sample collection guidance:\n"
                "- Use a clean, food-grade plastic or sterile bottle (500 mL recommended; 1 L for full SANS 241).\n"
                "- Rinse 3× with source water; leave ~1–2 cm headspace; seal tightly.\n"
                "- Keep chilled (≤10 °C) and deliver within 24 hours.\n"
                f"- Drop-off: {addr or 'please ask for the lab drop-off address.'}\n"
                f"- {hours or ''}\n"
                "Would you like me to recommend tests or prepare a quote?"
            )
            try:
                self.mem.save_turn(question, guidance)
            except Exception:
                pass
            return {"answer": guidance, "sources": [], "trace": _build_trace(intent="logistics")}

        # Escalation session flow
        if isinstance(session, ChatSession) and session.mode == "escalate":
            txt = q
            if session.escalate_field is None:
                session.escalate_field = "name"
            if session.escalate_field == "name":
                if not txt:
                    return {
                        "answer": "I'd be happy to arrange for a consultant to call you. Could you share your name?",
                        "sources": [],
                        "trace": _build_trace(intent="escalation"),
                    }
                session.escalate["name"] = txt
                session.escalate_field = "contact"
                return {
                    "answer": f"Thanks, {txt}. What's the best number or email address for our consultant to reach you?",
                    "sources": [],
                    "trace": _build_trace(intent="escalation"),
                }
            if session.escalate_field == "contact":
                EMAIL_RX = re.compile(
                    r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$", re.IGNORECASE)
                PHONE_RX = re.compile(r"^\+?[0-9 ()\-]{7,20}$")
                if not (EMAIL_RX.match(txt) or PHONE_RX.match(txt)):
                    return {
                        "answer": "I'll need a valid phone number or email address for the callback. Could you provide that?",
                        "sources": [],
                        "trace": _build_trace(intent="escalation"),
                    }
                session.escalate["contact"] = txt
                session.escalate_field = "time"
                return {
                    "answer": "Perfect. What would be the best time during business hours for them to contact you?",
                    "sources": [],
                    "trace": _build_trace(intent="escalation"),
                }
            if session.escalate_field == "time":
                session.escalate["time"] = txt
                session.escalate_field = "notes"
                return {
                    "answer": "Got it. Is there any specific question or topic you'd like the consultant to be prepared to discuss? (Optional)",
                    "sources": [],
                    "trace": _build_trace(intent="escalation"),
                }
            if session.escalate_field == "notes":
                if txt:
                    session.escalate["notes"] = txt
                # Summarize and ask for confirmation before handoff
                name = session.escalate.get("name", "")
                contact = session.escalate.get("contact", "")
                time = session.escalate.get("time", "during business hours")
                notes = session.escalate.get("notes", "")
                summary = [
                    "I'll ask a consultant to contact you. Please confirm these details:",
                    f"- Name: {name}",
                    f"- Contact: {contact}",
                    f"- Preferred time: {time}",
                ]
                if notes:
                    summary.append(f"- Notes: {notes}")
                summary.append(
                    "Reply 'yes' to confirm, or tell me what to change (name, contact, time, notes)."
                )
                session.escalate_field = "confirm"
                return {
                    "answer": "\n".join(summary),
                    "sources": [],
                    "trace": _build_trace(intent="escalation"),
                }
            if session.escalate_field == "confirm":
                if _is_acknowledgement(txt):
                    name = session.escalate.get("name", "")
                    time = session.escalate.get(
                        "time", "during business hours")
                    session.escalate_field, session.mode = "done", None
                    reply = (
                        f"Thank you, {name}. I've arranged for a consultant to call you back {time} on the next business day. "
                        "They'll be prepared to help with your questions. Is there anything else I can assist with in the meantime?"
                    )
                    try:
                        self.mem.save_turn(question, reply)
                    except Exception:
                        pass
                    return {
                        "answer": reply,
                        "sources": [],
                        "trace": _build_trace(intent="escalation"),
                    }
                # Determine which field to update
                low = (txt or "").lower()
                if "name" in low:
                    session.escalate_field = "name"
                    return {
                        "answer": "No problem — please share the correct name and surname.",
                        "sources": [],
                        "trace": _build_trace(intent="escalation"),
                    }
                if "contact" in low or "phone" in low or "email" in low:
                    session.escalate_field = "contact"
                    return {
                        "answer": "Sure — what's the best number or email address to reach you?",
                        "sources": [],
                        "trace": _build_trace(intent="escalation"),
                    }
                if "time" in low:
                    session.escalate_field = "time"
                    return {
                        "answer": "What would be the best time during business hours for them to contact you?",
                        "sources": [],
                        "trace": _build_trace(intent="escalation"),
                    }
                if "note" in low or "topic" in low:
                    session.escalate_field = "notes"
                    return {
                        "answer": "What should they be prepared to discuss? (Optional)",
                        "sources": [],
                        "trace": _build_trace(intent="escalation"),
                    }
                # Default: restart confirmation loop politely
                return {
                    "answer": "Got it. Which detail should I change: name, contact, time, or notes?",
                    "sources": [],
                    "trace": _build_trace(intent="escalation"),
                }

        # If quote flow already active: intercept pricing/TAT questions; else continue flow
        if isinstance(session, ChatSession):
            active_states = {
                "selection",
                "selection_confirm",
                "confirmation",
                "pdf",
                "collect_counts",
                "collect_name",
                "collect_company",
                "collect_email",
                "collect_extra",
            }
            active = (
                bool(getattr(session, "selected_tests", []))
                or bool(getattr(session, "last_options", []))
                or (getattr(session, "state", None) in active_states)
            )
            if active and getattr(session, "selected_tests", []):
                if _is_price_query(q) or _is_tat_query(q):
                    text = _summarize_selection(
                        session.selected_tests,
                        show_price=_is_price_query(q),
                        show_tat=_is_tat_query(q),
                    )
                    try:
                        self.mem.save_turn(question, text)
                    except Exception:
                        pass
                    return {"answer": text, "sources": [], "trace": _build_trace(intent="info")}
            if active:
                text = self.handle_quote_flow(q, session)
                try:
                    self.mem.save_turn(question, text)
                except Exception:
                    pass
                return {"answer": text, "sources": [], "trace": _build_trace(intent="quote")}

        # Build history/analysis
        history = self._get_history_snippet()
        analysis = self._understand_question(q, history)
        intent = analysis.get("intent", "info")

        # Kick off escalation if classified
        if intent == "escalation":
            if not session:
                session = ChatSession()
            session.mode = "escalate"
            session.escalate_field = "name"
            return {
                "answer": "I'd be happy to arrange for a consultant to call you. Could you share your name?",
                "sources": [],
                "trace": _build_trace(intent="escalation"),
            }

        # If clear quote intent (but not explicit price question), route to quote flow early
        lq = q.lower()
        looked_like_selection = bool(
            re.search(r"\b\d+(\s*(and|,)\s*\d+)*\b", lq))
        generic_test_interest = bool(re.search(r"\b(water|soil|fertiliser|fertilizer)\b", lq)) or (
            "test" in lq and len(lq.split()) <= 6
        )
        quote_intent_shortcircuit = (
            intent == "quote"
            or any(k in lq for k in ("quote", "pdf quote", "generate quote"))
            or looked_like_selection
            or generic_test_interest
        )
        price_keywords = any(k in q.lower()
                             for k in ("price", "cost", "how much", "fee"))
        if quote_intent_shortcircuit and not price_keywords:
            sess = session or ChatSession()
            text = self.handle_quote_flow(q, sess)
            try:
                self.mem.save_turn(question, text)
            except Exception:
                pass
            return {
                "answer": text,
                "sources": [],
                "trace": _build_trace(intent="quote", analysis=analysis),
            }

        # Retrieval
        aug_q = analysis.get("reformulated_query", q)
        docs = self._retrieve_enhanced_context(aug_q, analysis)
        context_text, _ = _format_context(docs)
        if extra_context and extra_context.strip():
            context_text = extra_context.strip() + "\n" + context_text

        # Relevance gate
        min_overlap = float(getattr(cfg, "MIN_KEYWORD_OVERLAP", 0.12))
        dist_max = float(getattr(cfg, "SIMILARITY_DISTANCE_MAX", 0.6))
        top_dist = top_similarity_distance(aug_q)
        ratio = _keyword_overlap_ratio(aug_q, context_text)

        insufficient_context = (
            bool(getattr(cfg, "ENABLE_NOT_SURE_FALLBACK", True))
            and (ratio < min_overlap)
            and (top_dist is None or top_dist > dist_max)
        )
        has_entity_matches = bool(analysis.get("entities", [])) and any(
            e.lower() in context_text.lower() for e in analysis.get("entities", [])
        )
        gated = insufficient_context and not has_entity_matches

        if gated:
            reply = (
                "I'm not completely sure I can answer that accurately in chat right now. "
                "I want to make sure I give you the most accurate information — could you confirm whether you're asking about pricing, sample drop-off, or would you like a consultant callback?"
            )
            try:
                self.mem.save_turn(question, reply)
            except Exception:
                pass
            return {
                "answer": reply,
                "sources": [],
                "trace": _build_trace(
                    intent=intent,
                    analysis=analysis,
                    gated=True,
                    keyword_overlap=ratio,
                    top_similarity_distance=top_dist,
                ),
            }

        # LLM answer
        hist = self._get_history_snippet()
        user_formatted = getattr(prompts, "USER_RAG", "{question}\n\n{context}").format(
            question=aug_q, context=context_text, history=hist or ""
        )
        system_msgs = [
            SystemMessage(
                content=getattr(prompts, "SYSTEM_RAG",
                                "Use only the given context.").strip()
            )
        ]
        sg = getattr(prompts, "STYLE_GUIDE", "").strip()
        if sg:
            system_msgs.append(SystemMessage(content=sg))
        pr = getattr(prompts, "PRINCIPLES_BRIEF", "").strip()
        if pr:
            system_msgs.append(SystemMessage(
                content=f"Operating principles:\n{pr}"))
        messages = [
            *system_msgs,
            *self._summary_message(),
            HumanMessage(content=user_formatted.strip()),
        ]

        try:
            resp = self.llm.invoke(messages)
            initial_answer = resp.content.strip() if hasattr(resp, "content") else str(resp)
        except Exception:
            initial_answer = ""

        refined_answer = self._refine_answer(
            q, initial_answer, context_text, hist)
        text = refined_answer

        # Sanitize output: remove markdown bold and hide TAT unless asked
        def _sanitize_output(user_q: str, ans: str) -> str:
            s = ans.replace("**", "")
            uq = (user_q or "").lower()
            wants_tat = ("tat" in uq) or ("turnaround" in uq)
            if not wants_tat:
                s = re.sub(
                    r"\s*[•\-\u2022]?\s*(Turnaround\s*Time\s*:\s*[^\n]+)",
                    "",
                    s,
                    flags=re.IGNORECASE,
                )
                s = re.sub(
                    r"\s*[•\-\u2022]?\s*TAT\s*\d+(?:\.\d+)?\s*days", "", s, flags=re.IGNORECASE
                )
                s = re.sub(r"\s{2,}", " ", s).replace(" ,", ",").strip()
            return s

        text = _sanitize_output(q, text)

        # Price guard
        if bool(getattr(cfg, "ENABLE_PRICE_GUARD", True)):
            if any(k in q.lower() for k in ["price", "cost", "fee"]) and _contains_price(text):
                ctx_has_price = bool(
                    re.search(r"(price[_\s-]?zar|R\s?\d)",
                              context_text, re.IGNORECASE)
                )
                if not ctx_has_price:
                    text = (
                        "I'm not completely sure I can answer that accurately in chat right now. "
                        "I want to make sure I give you the most accurate information — could you confirm the exact test so I can give you an accurate price?"
                    )

        try:
            self.mem.save_turn(question, text)
        except Exception:
            pass

        result = {"answer": text, "sources": []}
        result["trace"] = _build_trace(
            keyword_overlap=round(ratio, 3) if isinstance(
                ratio, float) else None,
            top_similarity_distance=top_dist,
            intent=intent,
            analysis=analysis,
            gated=False,
        )

        if (
            not result.get("answer")
            or not isinstance(result.get("answer"), str)
            or not result["answer"].strip()
        ):
            if quote_intent_shortcircuit and not price_keywords:
                sess = session or ChatSession()
                safe_text = self.handle_quote_flow(q, sess)
            else:
                safe_text = (
                    "Sorry — I couldn't generate a direct answer right now. "
                    "Could you rephrase that or tell me whether you'd like a quote, sample drop-off details, or to speak with a consultant?"
                )
            result["answer"] = safe_text

        return result

    # ---------------- LLM understanding / retrieval / refinement ----------------

    def _understand_question(self, question: str, history: str) -> Dict[str, Any]:
        if not bool(getattr(cfg, "ENABLE_QUESTION_UNDERSTANDING", True)):
            return {
                "reformulated_query": question,
                "intent": self._classify_intent_llm_safe(question, "").get("intent", "info"),
                "entities": [],
                "needs_clarification": False,
                "clarification_question": "",
            }

        try:
            sys_text = getattr(prompts, "QUESTION_UNDERSTANDING", "").strip() or (
                "Analyze the user's question and return JSON with keys: "
                "reformulated_query, intent (quote|info|logistics|escalation|other), "
                "entities (array of strings), needs_clarification (bool), clarification_question."
            )
            msgs = [
                SystemMessage(content=sys_text),
                HumanMessage(
                    content=f"Conversation history (recent):\n{(history or '')[-800:]}\n\nCurrent question:\n{question}\n\nReturn JSON only."
                ),
            ]
            resp = self.llm.invoke(msgs)
            raw = resp.content if hasattr(resp, "content") else str(resp)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                raw = m.group(0)
            obj = json.loads(raw)
            obj.setdefault("reformulated_query", question)
            obj.setdefault("intent", "info")
            obj.setdefault("entities", [])
            obj.setdefault("needs_clarification", False)
            obj.setdefault("clarification_question", "")
            return obj
        except Exception:
            return {
                "reformulated_query": question,
                "intent": self._classify_intent_llm_safe(question, history).get("intent", "info"),
                "entities": [],
                "needs_clarification": False,
                "clarification_question": "",
            }

    def _retrieve_enhanced_context(self, query: str, analysis: Dict[str, Any]) -> List[Any]:
        docs: List[Any] = []
        try:
            base_docs = _retrieve_docs(self.retriever, query)
            docs.extend(base_docs or [])
        except Exception:
            pass

        rq = (analysis or {}).get("reformulated_query")
        if rq and rq != query:
            try:
                alt_docs = _retrieve_docs(self.retriever, rq)
                for d in alt_docs or []:
                    if d not in docs:
                        docs.append(d)
            except Exception:
                pass

        ents = (analysis or {}).get("entities") or []
        if ents:
            ent_q = " OR ".join(map(str, ents)).strip()
            if ent_q:
                try:
                    ent_docs = _retrieve_docs(self.retriever, ent_q)
                    for d in ent_docs or []:
                        if d not in docs:
                            docs.append(d)
                except Exception:
                    pass

        return docs[: int(self._config.get("top_k", 4))]

    def _generate_clarification_question(
        self, question: str, analysis: Dict[str, Any], history: str
    ) -> str:
        try:
            sys_content = (
                "Ask one short, conversational follow-up to clarify the user's request. "
                "Be specific and friendly. Offer options if helpful. Avoid saying 'I don't understand'."
            )
            unclear = (
                analysis.get(
                    "clarification_question") or "Which test or outcome should I focus on?"
            )
            user_content = f"History:\n{(history or '')[-400:]}\n\nWhat's unclear:\n{unclear}\n\nWrite a single clarifying question (1–2 sentences)."
            resp = self.conversation_llm.invoke(
                [SystemMessage(content=sys_content),
                 HumanMessage(content=user_content)]
            )
            text = (resp.content if hasattr(
                resp, "content") else str(resp)).strip()
            return (
                text.splitlines()[0][:220]
                if text
                else "Could you share which specific test you have in mind?"
            )
        except Exception:
            return "Could you share which specific test you have in mind?"

    def _refine_answer(
        self, question: str, initial_answer: str, context_text: str, history: str
    ) -> str:
        if not bool(getattr(cfg, "ENABLE_ANSWER_REFINEMENT", True)):
            return initial_answer
        try:
            sys_text = getattr(prompts, "ANSWER_REFINEMENT", "").strip() or (
                "Refine the assistant's answer to be warm, professional, and accurate to the context. "
                "Do not add facts not supported by context."
            )
            user_text = (
                f"Recent history:\n{(history or '')[-400:]}\n\n"
                f"User question:\n{question}\n\n"
                f"Original answer:\n{initial_answer}\n\n"
                f"Context (facts to stay within):\n{context_text[:1000]}\n\n"
                "Return only the refined answer."
            )
            resp = self.llm.invoke(
                [SystemMessage(content=sys_text),
                 HumanMessage(content=user_text)]
            )
            refined = resp.content if hasattr(resp, "content") else str(resp)
            return (refined or initial_answer).strip()
        except Exception:
            return initial_answer

    # --------------------------- Structured (JSON) quote ---------------------------

    def structured_quote(self, question: str) -> Dict[str, Any]:
        if _is_location_query(question):
            addr = getattr(cfg, "DROP_OFF_ADDRESS", "")
            obj = {"tests": [], "total_price_ZAR": 0.0,
                   "notes": addr, "next_step": "proceed"}
            try:
                self.mem.save_turn(question, json.dumps(obj))
            except Exception:
                pass
            return obj

        aug_q, history = self._augment_follow_up(question)
        docs = _retrieve_docs(self.retriever, aug_q)
        context_text, _ = _format_context(docs)

        user_formatted = getattr(
            prompts, "USER_STRUCTURED_QUOTE", "{question}\n\n{context}"
        ).format(question=aug_q, context=context_text, history=history or "")
        system_msgs = [
            SystemMessage(
                content=getattr(
                    prompts, "SYSTEM_STRUCTURED_QUOTE", "Return valid JSON only."
                ).strip()
            )
        ]
        qp = getattr(prompts, "QUOTE_PRINCIPLES_BRIEF", "").strip()
        if qp:
            system_msgs.append(SystemMessage(
                content=f"Quoting principles:\n{qp}"))
        messages = [
            *system_msgs,
            *self._summary_message(),
            HumanMessage(content=user_formatted.strip()),
        ]

        resp = self.llm.invoke(messages)
        raw = resp.content if hasattr(resp, "content") else str(resp)

        obj = _ensure_json(raw)
        obj = _post_validate_quote(obj)

        tests = obj.get("tests") or []
        invalid = (
            not isinstance(tests, list)
            or len(tests) == 0
            or any(
                (not isinstance(t, dict))
                or (not t.get("id"))
                or math.isnan(_coerce_number(t.get("price_ZAR")))
                for t in tests
            )
        )

        if invalid:
            valid_options = [
                o
                for o in _docs_to_options(docs, limit=5)
                if o.get("price_ZAR") not in (None, "", 0)
            ]
            if len(valid_options) == 1:
                _p = float(_coerce_number(valid_options[0]["price_ZAR"]))
                obj = {
                    "tests": [{**valid_options[0], "price_ZAR": _p}],
                    "total_price_ZAR": _p,
                    "notes": "",
                    "next_step": "collect_client_details",
                    "clarification_required": False,
                    "options": [],
                }
            else:
                query_lower = aug_q.lower()
                best_match = None
                for opt in valid_options:
                    name_lower = (opt.get("name") or "").lower()
                    if name_lower in query_lower or any(
                        word in query_lower for word in name_lower.split()
                    ):
                        best_match = opt
                        break
                if best_match:
                    _p = float(_coerce_number(best_match.get("price_ZAR")))
                    obj = {
                        "tests": [{**best_match, "price_ZAR": _p}],
                        "total_price_ZAR": _p,
                        "notes": "",
                        "next_step": "collect_client_details",
                        "clarification_required": False,
                        "options": [],
                    }
                else:
                    options = _docs_to_options(docs, limit=5)
                    obj = {
                        "tests": [],
                        "total_price_ZAR": 0.0,
                        "notes": "",
                        "next_step": "unknown",
                        "clarification_required": True,
                        "clarification_question": "Please confirm the exact test you need so I can give you an accurate price.",
                        "options": options,
                    }
        else:
            obj["clarification_required"] = False
            obj["next_step"] = "collect_client_details"

        # Normalize numeric fields to floats
        try:
            if isinstance(obj.get("tests"), list):
                for t in obj["tests"]:
                    if isinstance(t, dict) and "price_ZAR" in t:
                        t["price_ZAR"] = float(
                            _coerce_number(t.get("price_ZAR")))
            if "total_price_ZAR" in obj:
                obj["total_price_ZAR"] = float(
                    _coerce_number(obj.get("total_price_ZAR")))
        except Exception:
            pass

        try:
            self.mem.save_turn(question, json.dumps(obj))
        except Exception:
            pass

        if not bool(getattr(cfg, "RETURN_DEBUG_TRACE", False)):
            obj.pop("_meta", None)
        else:
            obj.setdefault("_meta", {})
            obj["_meta"].update(
                {"prompt_version": getattr(cfg, "PROMPT_VERSION", "unknown")})
        return obj


# --------------------------- Quote text renderer (for tests/UI) ---------------------------


def _render_quote_text(obj: Dict[str, Any]) -> str:
    """Render a human‑readable quote summary from a structured quote object.

    Expectations (per tests):
    - Include ZAR formatting like R303.00
    - When clarification_required is True, list numbered options with '• TAT X days'
    - For multiple selected tests, render bullet lines with ', TAT X days'
    - Include validity text and a prompt to proceed to PDF
    """
    try:
        tests = obj.get("tests") or []
        clarify = bool(obj.get("clarification_required"))
        options = obj.get("options") or []
        lines: list[str] = []

        def fmt_price(x) -> str:
            try:
                return f"R{float(x):,.2f}"
            except Exception:
                return "R0.00"

        if clarify and options:
            lines.append("Please select an option to proceed:")
            for i, opt in enumerate(options, 1):
                name = str(opt.get("name", "Option"))
                price = fmt_price(opt.get("price_ZAR", 0))
                tat = opt.get("turnaround_days", None)
                tat_str = f" • TAT {int(tat)} days" if isinstance(
                    tat, (int, float)) else ""
                lines.append(f"{i}. {name} — {price}{tat_str}")
            lines.append("\nPlease select an option by number.")
            return "\n".join(lines)

        if tests:
            # Selected tests summary
            lines.append("Quote:")
            for t in tests:
                name = str(t.get("name", "Test"))
                price = fmt_price(t.get("price_ZAR", 0))
                tat = t.get("turnaround_days", None)
                tail = f", TAT {int(tat)} days" if isinstance(
                    tat, (int, float)) else ""
                lines.append(f"- {name} — {price}{tail}")
            total = obj.get("total_price_ZAR", 0)
            lines.append(f"Total R{float(total):,.2f}")
            lines.append("Valid 14 days.")
            lines.append("Would you like a formal PDF quote?")
            return "\n".join(lines)

        # Fallback
        return "I need a bit more detail to prepare an accurate quote. Please select from the options or specify the test."
    except Exception:
        return "Unable to render quote at the moment. Please try again."


def _format_price_zar(val) -> str:
    """Format numeric values as South African Rand with 2 decimals; passthrough otherwise."""
    try:
        return f"R{float(val):,.2f}"
    except Exception:
        return str(val)
