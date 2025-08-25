"""
RAG chains for the Highveld Biotech chatbot.

Provides two public entry points:
- RAGService.answer_question(question: str) -> dict   # prose answer + sources
- RAGService.structured_quote(question: str) -> dict  # strict JSON for quotes/booking

This module:
- Loads a retriever from the persisted FAISS index.
- Formats retrieved docs into a compact {context} with HVB IDs.
- Uses prompt constants from app.prompts.
- Calls an LLM (ChatOpenAI) with low temperature for factuality.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Tuple

from app.retriever import get_retriever, top_similarity_distance
import app.prompts as prompts
try:
    from app.memory import MemoryService  # type: ignore
except Exception:
    # Fallback shim if memory module or class is unavailable
    class MemoryService:  # type: ignore
        def save_turn(self, user_message: str, ai_message: str) -> None:
            pass

        def load_summary(self) -> str:
            return ""

        def reset(self) -> None:
            pass
# Import config module and read knobs with safe fallbacks
import app.config as cfg

# LangChain LLM interface (OpenAI chat)
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# -------- Helpers to parse CSVLoader page_content into key/value -------- #

def _lines_to_kv(lines: List[str]) -> Dict[str, str]:
    """Turn 'key: value' lines into a dict, tolerant to extra colons in values."""
    out: Dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if ":" not in line:
            # If a line has no colon, treat it as a freeform note appended to 'notes'
            out["notes"] = (out.get("notes", "") + " " + line).strip()
            continue
        k, v = line.split(":", 1)
        out[k.strip().lower()] = v.strip()
    return out


def _parse_doc_content(text: str) -> Dict[str, str]:
    """
    CSVLoader typically produces page_content like:
      'id: HVB-0001\ncategory: panel\ntest_name: Core Water Analysis\nprice_ZAR: 1700.0\n...'
    We convert it to a dict with lowercased keys.
    """
    lines = text.splitlines()
    return _lines_to_kv(lines)


def _extract_id(doc) -> str:
    """Best-effort extraction of the HVB ID from the document content."""
    d = _parse_doc_content(doc.page_content)
    # Prefer 'id' field if present
    id_val = d.get("id")
    if id_val:
        return id_val
    # Fallback: try to find 'HVB-####' in the content
    m = re.search(r"\bHVB-\d{4}\b", doc.page_content)
    if m:
        return m.group(0)
    # Last resort: try row number from metadata
    row = doc.metadata.get("row")
    if row is not None:
        return f"ROW-{row}"
    return "UNKNOWN-ID"


def _short_snippet(doc) -> str:
    """Build a short human-friendly snippet using key fields if available."""
    d = _parse_doc_content(doc.page_content)
    name = d.get("test_name") or d.get("category") or ""
    price = d.get("price_zar") or d.get("price") or ""
    tat = d.get("turnaround_days") or d.get("tat") or ""
    addr = d.get("address", "")
    category = d.get("category", "")
    bits = []
    if category == "drop_off":
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
            try:
                p = float(str(price).replace(',', '').replace('R', '').strip())
                bits.append(f"price {p:,.2f} ZAR (R{p:,.2f})")
            except Exception:
                bits.append(f"price_ZAR {price}")
        if tat:
            bits.append(f"TAT {tat} days")
    return " — ".join(bits)[:220]


def _format_context(docs: List[Any]) -> Tuple[str, List[str]]:
    """
    Turn retrieved Documents into a compact context block without IDs.
    Return (context_text, ordered_ids) where ordered_ids is kept only for internal tracing.
    """
    lines: List[str] = []
    ids: List[str] = []
    for doc in docs:
        hvb_id = _extract_id(doc)
        ids.append(hvb_id)
        snippet = _short_snippet(doc)
        lines.append(f"• {snippet}")
    ctx = "\n".join(lines)
    # Trim overly long contexts just in case
    if len(ctx) > 3000:
        ctx = ctx[:3000] + "\n…"
    return ctx, ids


def _tokenize(text: str) -> List[str]:
    t = (text or "").lower()
    # Keep alphanumerics as tokens; split on non-letters/numbers
    tokens = re.split(r"[^a-z0-9]+", t)
    # Filter short/common stop words
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
        "do", "does", "can", "i", "you", "we", "it", "at", "by", "be", "as", "from", "that",
        "this", "what", "how", "much", "many", "time", "turnaround", "tat", "price", "cost",
    }
    return [w for w in tokens if len(w) >= 3 and w not in stop]


def _keyword_overlap_ratio(question: str, context_text: str) -> float:
    q_words = set(_tokenize(question))
    if not q_words:
        return 0.0
    ctx_words = set(_tokenize(context_text))
    overlap = q_words & ctx_words
    return len(overlap) / max(1, len(q_words))


def _not_sure_answer(question: str) -> str:
    q = (question or "").strip()
    base = (
        "I'm not completely sure based on the information I have here. "
        "I don't want to guess and give you an inaccurate answer."
    )
    # A helpful, anticipatory follow-up tailored to common intents
    follow = "Could you clarify what test or detail you need (e.g., test name, price, or turnaround time)?"
    if any(k in q.lower() for k in ["price", "cost", "fee"]):
        follow = "Which test or panel are you asking about so I can confirm the exact price for you?"
    elif any(k in q.lower() for k in ["turnaround", "tat", "how long", "time"]):
        follow = "Which specific test are you asking about so I can confirm the turnaround time?"
    elif any(k in q.lower() for k in ["address", "drop", "location", "deliver"]):
        # Let the location override handle it; still provide a helpful nudge
        follow = "Are you asking for the sample drop-off address or courier details?"
    return f"{base} {follow}"


def _contains_price(text: str) -> bool:
    """Heuristic: detect price-like patterns R1234, 1,234.00, etc."""
    t = (text or "")
    return bool(re.search(r"(R\s?\d[\d,]*\.?\d{0,2}|\b\d{3,}[\.,]\d{2}\b)", t))


def _is_location_query(q: str) -> bool:
    ql = (q or "").lower()
    # Keywords indicating an address/drop-off/location query
    return any(k in ql for k in [
        "where can i drop off",
        "drop off",
        "drop-off",
        "address",
        "location",
        "physical address",
        "where do i deliver",
        "where can i deliver",
        "where to drop",
    ])


def _pick_model() -> Tuple[str, float, int]:
    """Pull model name, temperature, and top_k from config if present, else use defaults."""
    # Provide safe defaults if these aren't defined in config.py
    model = getattr(cfg, "LLM_MODEL", "gpt-4o-mini")
    temperature = getattr(cfg, "LLM_TEMPERATURE", 0.1)
    top_k = getattr(cfg, "TOP_K", 4)
    return str(model), float(temperature), int(top_k)


# Prefer retriever.invoke (new API); fall back to get_relevant_documents
_def_invoke = getattr(object, "__name__", None)  # no-op to keep linters calm


def _retrieve_docs(retriever, query: str):
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    return retriever.get_relevant_documents(query)


def _ensure_json(s: str) -> Dict[str, Any]:
    """Parse a JSON string safely; if it fails, raise a ValueError with context."""
    s = s.strip()
    # Some models inadvertently wrap JSON in ```json fences; strip them.
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception as e:
        raise ValueError(
            f"Model did not return valid JSON.\nRaw output:\n{s}") from e


def _coerce_number(x: Any) -> float:
    """Coerce stringified numbers to float; ignore non-numeric gracefully."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, str):
        # Remove thousands commas and currency symbols if any leaked in
        x = x.replace(",", "").replace("R", "").strip()
        try:
            return float(x)
        except ValueError:
            return math.nan
    return math.nan


def _post_validate_quote(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Light validation for structured quotes:
    - Ensure tests[*].price_ZAR and .turnaround_days are numeric (or NaN if missing)
    - If total_price_ZAR missing or NaN, compute sum of item prices
    - Clamp next_step to allowed values
    """
    tests = obj.get("tests", [])
    if not isinstance(tests, list):
        tests = []

    for t in tests:
        if not isinstance(t, dict):
            continue
        t["price_ZAR"] = _coerce_number(t.get("price_ZAR"))
        t["turnaround_days"] = _coerce_number(t.get("turnaround_days"))

    total = obj.get("total_price_ZAR")
    total_num = _coerce_number(total)
    if math.isnan(total_num):
        total_num = sum(p for p in (_coerce_number(t.get("price_ZAR"))
                        for t in tests) if not math.isnan(p))
    obj["total_price_ZAR"] = round(total_num, 2)

    allowed = {"book_call", "proceed", "unknown"}
    step = obj.get("next_step", "unknown")
    if step not in allowed:
        obj["next_step"] = "unknown"

    # Ensure basic keys exist
    obj.setdefault("notes", "")
    obj["tests"] = tests
    return obj


# -------------------------- Public Service Class -------------------------- #

class RAGService:
    """High-level service that encapsulates retrieval + prompting."""

    def __init__(
        self,
        llm_model: str | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        memory: MemoryService | None = None,
        session_id: str | None = None,
    ):
        default_model, default_temp, default_topk = _pick_model()
        model_name = llm_model or default_model
        temp = default_temp if temperature is None else float(temperature)
        k = default_topk if top_k is None else int(top_k)
        self.llm = ChatOpenAI(model=model_name, temperature=temp)
        self.retriever = get_retriever(top_k=k)
        self.mem = memory or MemoryService(session_id=session_id)
        # Keep for trace/debug
        self._config = {"model": model_name, "temperature": temp, "top_k": k}

    def _summary_message(self) -> List[SystemMessage]:
        try:
            summary = (self.mem.load_summary() or "").strip()
        except Exception:
            summary = ""
        if summary:
            return [SystemMessage(content=f"Conversation summary so far (for context, not as facts):\n{summary}")]
        return []

    # ---- Mode 1: Prose answer + citations ---- #
    def answer_question(self, question: str, *, extra_context: str | None = None) -> Dict[str, Any]:
        """Return a dict with {answer, sources, trace}."""
        # 0) Canonical override for drop-off/address/location queries
        if _is_location_query(question):
            addr = getattr(cfg, "DROP_OFF_ADDRESS", "")
            if addr:
                text = addr.strip()
                try:
                    self.mem.save_turn(question, text)
                except Exception:
                    pass
                return {
                    "answer": text,
                    "sources": [],
                    "trace": {"override": "DROP_OFF_ADDRESS"},
                }

        # 1) Retrieve
        docs = _retrieve_docs(self.retriever, question)

        # 2) Format context
        context_text, ordered_ids = _format_context(docs)
        if extra_context and extra_context.strip():
            # Prepend caller-provided context so it's prominent
            context_text = extra_context.strip() + "\n" + context_text

        # 2.4) Similarity-distance gate (if vector store exposes scores)
        dist_max = float(getattr(cfg, "SIMILARITY_DISTANCE_MAX", 0.6))
        top_dist = top_similarity_distance(question)

        # 2.5) Confidence check to reduce hallucinations
        min_overlap = float(getattr(cfg, "MIN_KEYWORD_OVERLAP", 0.12))
        enable_fallback = bool(getattr(cfg, "ENABLE_NOT_SURE_FALLBACK", True))
        ratio = _keyword_overlap_ratio(question, context_text)
        # If extra_context is provided and clearly relevant (high overlap), do not gate.
        has_helpful_extra = bool(extra_context and ratio >= (min_overlap * 1.2))
        gated = (
            enable_fallback
            and not has_helpful_extra
            and (ratio < min_overlap)
            and (top_dist is None or top_dist > dist_max)
        )
        if gated:
            text = _not_sure_answer(question)
            try:
                self.mem.save_turn(question, text)
            except Exception:
                pass
            return {
                "answer": text,
                "sources": [],
                "trace": {
                    "retrieved_ids": ordered_ids,
                    "context_preview": context_text,
                    "keyword_overlap": round(ratio, 3),
                    "gated": True,
                    "top_similarity_distance": top_dist,
                    "distance_max": dist_max,
                },
            }

        # 3) Build messages
        user_formatted = getattr(prompts, 'USER_RAG', '{question}\n\n{context}').format(
            question=question, context=context_text)
        messages = [
            SystemMessage(content=getattr(prompts, 'SYSTEM_RAG',
                          'You are a helpful assistant. Use only the given context.').strip()),
            *self._summary_message(),
            HumanMessage(content=user_formatted.strip()),
        ]

        # 4) Call model
        resp = self.llm.invoke(messages)
        text = resp.content.strip() if hasattr(resp, "content") else str(resp)

        # 4.5) Price guard: if the question suggests pricing and output includes a price
        # but retrieved context has no obvious price snippets, prefer to ask for clarity.
        if bool(getattr(cfg, "ENABLE_PRICE_GUARD", True)):
            if any(k in question.lower() for k in ["price", "cost", "fee"]) and _contains_price(text):
                ctx_has_price = bool(re.search(r"price[_\s-]?zar|R\s?\d", context_text, re.IGNORECASE))
                if not ctx_has_price:
                    text = _not_sure_answer(question)

        # 5) Save memory; do not return sources to user
        try:
            self.mem.save_turn(question, text)
        except Exception:
            pass

        return {
            "answer": text,
            "sources": [],
            "trace": {
                "retrieved_ids": ordered_ids,
                "context_preview": context_text,
                "model": getattr(self.llm, "model_name", getattr(self.llm, "model", None)),
                "prompt_version": getattr(cfg, "PROMPT_VERSION", "unknown"),
            },
        }

    def stream_answer(self, question: str, *, extra_context: str | None = None):
        """Yield answer text incrementally (token/chunk streaming).
        Note: streaming does not auto-save to memory; callers should
        collect the final text and call MemoryService.save_turn(question, final_text).
        """
        # Override stream for location queries: yield the canonical address and return
        if _is_location_query(question):
            addr = getattr(cfg, "DROP_OFF_ADDRESS", "")
            yield addr
            return
        # 1) Retrieve (non-streaming)
        docs = _retrieve_docs(self.retriever, question)
        context_text, _ = _format_context(docs)
        if extra_context and extra_context.strip():
            context_text = extra_context.strip() + "\n" + context_text

        # Confidence gating in streaming mode: if weak context, emit not-sure text and stop
        min_overlap = float(getattr(cfg, "MIN_KEYWORD_OVERLAP", 0.12))
        enable_fallback = bool(getattr(cfg, "ENABLE_NOT_SURE_FALLBACK", True))
        ratio = _keyword_overlap_ratio(question, context_text)
        dist_max = float(getattr(cfg, "SIMILARITY_DISTANCE_MAX", 0.6))
        top_dist = top_similarity_distance(question)
        if enable_fallback and (ratio < min_overlap) and (top_dist is None or top_dist > dist_max):
            yield _not_sure_answer(question)
            return

        # 2) Build messages
        user_formatted = getattr(prompts, 'USER_RAG', '{question}\n\n{context}').format(
            question=question, context=context_text)
        messages = [
            SystemMessage(content=getattr(prompts, 'SYSTEM_RAG',
                          'You are a helpful assistant. Use only the given context.').strip()),
            *self._summary_message(),
            HumanMessage(content=user_formatted.strip()),
        ]

        # 3) Stream model output
        for chunk in self.llm.stream(messages):
            piece = getattr(chunk, "content", None)
            if piece:
                yield piece

    # ---- Mode 2: JSON-only quote/booking ---- #
    def structured_quote(self, question: str) -> Dict[str, Any]:
        """Return a dict parsed from JSON (tests, total_price_ZAR, notes, next_step)."""
        # For location-only questions, return a minimal JSON with notes set to the address
        if _is_location_query(question):
            addr = getattr(cfg, "DROP_OFF_ADDRESS", "")
            obj = {
                "tests": [],
                "total_price_ZAR": 0.0,
                "notes": addr,
                "next_step": "proceed",
            }
            try:
                self.mem.save_turn(question, json.dumps(obj))
            except Exception:
                pass
            return obj
        # 1) Retrieve
        docs = _retrieve_docs(self.retriever, question)

        # 2) Format context
        context_text, _ = _format_context(docs)

    # 3) Build messages for strict JSON
        user_formatted = getattr(prompts, 'USER_STRUCTURED_QUOTE', '{question}\n\n{context}').format(
            question=question, context=context_text)
        messages = [
            SystemMessage(content=getattr(
                prompts, 'SYSTEM_STRUCTURED_QUOTE', 'Return valid JSON only.').strip()),
            *self._summary_message(),
            HumanMessage(content=user_formatted.strip()),
        ]

        # 4) Call model
        resp = self.llm.invoke(messages)
        raw = resp.content if hasattr(resp, "content") else str(resp)

        # 5) Parse + post-validate + save memory
        obj = _ensure_json(raw)
        obj = _post_validate_quote(obj)
        try:
            self.mem.save_turn(question, json.dumps(obj))
        except Exception:
            pass
        # Annotate trace-like details inline for debugging consumers.
        obj.setdefault("_meta", {})
        obj["_meta"].update({
            "prompt_version": getattr(cfg, "PROMPT_VERSION", "unknown"),
        })
        return obj


# -------------------------- Small Utilities -------------------------- #

def _extract_ids_from_answer(answer_text: str) -> List[str]:
    """
    Try to extract 'Sources: HVB-0001, HVB-0007' from the model's prose output.
    If not found, return an empty list (caller will fall back to retrieved IDs).
    """
    # Find the 'Sources:' line
    m = re.search(r"(?im)^\s*Sources?:\s*(.+)$", answer_text)
    if not m:
        return []
    tail = m.group(1)
    # Extract HVB-#### patterns
    ids = re.findall(r"\bHVB-\d{4}\b", tail)
    # Deduplicate preserving order
    seen = set()
    uniq: List[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq
