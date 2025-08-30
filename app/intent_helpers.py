"""
Centralized intent detection helpers to avoid duplication across files.
"""

from __future__ import annotations

import re
import json
from typing import List, Dict

# ---------------------------
# Regex patterns (robust)
# ---------------------------

_ACK_RX = re.compile(
    r"^(?:yes(?: please)?|yeah|yep|ok(?:ay)?|please|sure|go ahead|do it|confirm(?: & continue)?|sounds good|that(?:'s| is) fine|proceed)\.?$",
    re.IGNORECASE,
)

_NEG_RX = re.compile(
    r"^(?:no|nope|nah|cancel|stop|quit|don'?t|do not|negative)\.?$",
    re.IGNORECASE,
)

_GREETING_RX = re.compile(
    r"^\s*(?:hi|hello|hey|good (?:morning|afternoon|evening|day)|how (?:are|r) (?:you|u))\b",
    re.IGNORECASE,
)

_ESCALATE_RX = re.compile(
    r"(?:consultant|human|call (?:me|back)|phone\s*me|speak to (?:someone|a human)|someone to call|contact me|escalate)",
    re.IGNORECASE,
)

# NOTE: QUOTE intent should map to price/estimate language.
# If you *want* generic "test"/"testing" to trigger quotes, keep them here;
# otherwise it will over-trigger. I’ve kept them (as in your original) to avoid breaking flows.
_QUOTE_RX = re.compile(
    r"\b(?:quote|quotation|please quote|quote please|send (?:me )?a quote|pricing for|price for|cost of|how much (?:is|for)|estimate|can i get a quote|quote on|test|tests|testing|analysis|assay|determinand|sample|lab|compliance|standard|soil)\b",
    re.IGNORECASE,
)

_SAMPLE_RX = re.compile(
    r"(?:how (?:do|to) (?:i )?(?:collect|take|get|prepare).*sample|"
    r"prepare (?:a |the )?(?:water|soil)?\s*sample|"
    r"sample (?:collection|bottle|kit)|get a (?:water|soil) sample)",
    re.IGNORECASE,
)

_PDF_RX = re.compile(
    r"(?:\b(?:generate|prepare|make)\b.*\b(?:pdf)\b.*\b(?:quote)\b)|\bformal pdf quote\b|\bpdf quote\b",
    re.IGNORECASE,
)

# Technical / standards mentions
_TECHNICAL_RX = re.compile(
    r"(?:methodolog(?:y|ies)|spectrophotometry|gravimetric|titration|acid digestion|cv[-\s]*af|hg|sans\s*241|microbiological|chemical determinand|environmental monitoring|compliance|accreditation)",
    re.IGNORECASE,
)

# ---------------------------
# Public helpers (stable API)
# ---------------------------


def _is_greeting(s: str) -> bool:
    return bool(_GREETING_RX.search((s or "").strip()))


def _is_acknowledgement(q: str) -> bool:
    return bool(_ACK_RX.match((q or "").strip()))


def _is_negative(q: str) -> bool:
    return bool(_NEG_RX.match((q or "").strip()))


def _is_escalation_request(q: str) -> bool:
    return bool(_ESCALATE_RX.search((q or "").strip()))


def _is_quote_intent(q: str) -> bool:
    return bool(_QUOTE_RX.search((q or "").strip()))


def _is_sample_collection_query(q: str) -> bool:
    return bool(_SAMPLE_RX.search((q or "").strip()))


def _is_pdf_request(q: str) -> bool:
    return bool(_PDF_RX.search((q or "").strip()))


def _is_price_query(q: str) -> bool:
    s = (q or "").lower()
    return any(k in s for k in ("price", "pricing", "cost", "how much", "fee"))


def _is_tat_query(q: str) -> bool:
    s = (q or "").lower()
    return ("tat" in s) or ("turnaround" in s) or ("how long" in s) or ("lead time" in s)


def _is_location_query(q: str) -> bool:
    s = (q or "").lower()
    return any(
        k in s
        for k in ["address", "drop-off", "drop off", "where to deliver", "location", "dropoff"]
    )


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9\-]+", (text or "").lower())


def _get_test_names() -> set[str]:
    # Placeholder; plug in your retriever-backed test names if/when available
    return set()


def _classify_intent_llm(text: str, history: str = "") -> dict:
    # Placeholder for an LLM-based intent layer if needed
    return {"intent": "info", "entities": []}


def _is_affirm(q: str) -> bool:
    """Return True for short affirmative replies."""
    if not q:
        return False
    s = _normalize(q)
    if s in AFFIRM:
        return True
    return bool(_ACK_RX.match((q or "").strip()))


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


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


def _lines_to_kv(lines: List[str]) -> Dict[str, str]:
    """
    Convert a list of 'key: value' lines into a dict with lowercased keys.
    Lines without ':' are ignored. Later duplicates overwrite earlier ones.
    """
    out: Dict[str, str] = {}
    for line in lines:
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        out[key.strip().lower()] = val.strip()
    return out


def _parse_doc_content(text: str) -> Dict[str, str]:
    """
    CSVLoader typically produces page_content like:
      'id: HVB-0001\ncategory: panel\ntest_name: Core Water Analysis\nprice_ZAR: 1700.0\n...'
    We convert it to a dict with lowercased keys.
    """
    return _lines_to_kv((text or "").splitlines())


def _retrieve_docs(retriever, query: str):
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    return []


def _maybe_extract_answer_field(response: str) -> str:
    # Placeholder passthrough
    return response


def _extract_recent_topic(history: str) -> str:
    words = (history or "").split()[-10:]
    return " ".join(words) if words else ""


def _question_sentence_from(history: str) -> str:
    # Naive: last sentence containing '?'
    for sentence in reversed((history or "").split(".")):
        if "?" in sentence:
            return sentence.strip()
    return ""


def _sentiment_score(text: str) -> float:
    """Very lightweight sentiment scorer for tests and gentle gating.

    - Tokenizes by alphanumerics (drops punctuation), making "great!" -> "great".
    - Assigns stronger weights so short positive phrases exceed 0.5 as tests expect.
    """
    pos = {
        "good",
        "great",
        "excellent",
        "awesome",
        "happy",
        "thanks",
        "thank",
        "appreciate",
        "fantastic",
        "wonderful",
        "perfect",
    }
    neg = {"bad", "terrible", "frustrated", "angry", "useless", "awful", "hate"}
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    score = 0.0
    for w in toks:
        if w in pos:
            score += 0.4
        elif w in neg:
            score -= 0.6
    # Mild normalization for long inputs
    score = max(-1.0, min(1.0, score))
    return score


def _quick_suggestions() -> str:
    return (
        "Popular topics: Water tests • Fertiliser/Soil tests • Other chemical tests.\n"
        "Ask about pricing and turnaround time, drop-off location and hours, sample preparation, or request a quote."
    )


def _ensure_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _post_validate_quote(obj: dict) -> dict:
    required_keys = ["tests", "total_price_ZAR", "notes"]
    for k in required_keys:
        if k not in obj:
            obj[k] = [] if k == "tests" else (0.0 if k == "total_price_ZAR" else "")
    return obj


def _coerce_number(val) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _docs_to_options(docs, limit: int = 5) -> list:
    options = []
    for doc in (docs or [])[:limit]:
        content = getattr(doc, "page_content", "") or ""
        name = "Unknown Test"
        price = "0"
        if "test_name: " in content:
            try:
                name = content.split("test_name: ", 1)[1].split("\n", 1)[0].strip()
            except Exception:
                pass
        if "price_ZAR: " in content:
            try:
                price = content.split("price_ZAR: ", 1)[1].split("\n", 1)[0].strip()
            except Exception:
                pass
        options.append({"name": name, "price_ZAR": price})
    return options


def _is_technical_question(q: str) -> bool:
    return bool(_TECHNICAL_RX.search((q or "").strip()))


def _extract_standards(text: str) -> List[str]:
    standards: List[str] = []
    t = text or ""
    if "SANS 241" in t.upper().replace("\u00a0", " "):
        standards.append("SANS 241")
    return standards
