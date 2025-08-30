"""
Stateful quote flow helper. Keeps logic self-contained to avoid circular imports.

RAGService uses:
    from app.quote_flow import handle_quote_flow

This module expects:
    - ChatSession, SelectedTest dataclasses in app.state
    - A retriever with .invoke(query) or .get_relevant_documents(query)
"""

from __future__ import annotations

import re
from typing import List

from app.state import ChatSession, SelectedTest
from app.intent_helpers import (
    _parse_doc_content,
    _retrieve_docs,
    _is_acknowledgement,
    _is_negative,
)

# ---------------- small local helpers (duplicated on purpose to avoid cycles) ----------------


def _match_user_choices(user_text: str, options: List[SelectedTest]) -> List[SelectedTest]:
    """Match numbers or names (fuzzy) from user_text to options.

    - Accepts inputs like: "2", "2 please", "1 3", "uv disinfection", "nitrate test".
    - Uses numeric matching first; then fuzzy token overlap against option names.
    """
    txt = (user_text or "").lower().split("?")[0].strip()
    chosen: List[SelectedTest] = []

    # 1) Numbers like "1 3" or "2, please"
    nums = set(re.findall(r"\b\d+\b", txt))
    for idx, opt in enumerate(options):
        if str(idx + 1) in nums:
            chosen.append(opt)

    # 2) Fuzzy token overlap on names
    def _tokens(s: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", (s or "").lower())

    q_tokens = set(_tokens(txt))
    if q_tokens:
        best: List[tuple[float, SelectedTest]] = []
        for opt in options:
            name = (opt.name or "").lower()
            o_tokens = set(_tokens(name))
            if not o_tokens:
                continue
            inter = len(q_tokens & o_tokens)
            ratio = inter / max(1, len(q_tokens))
            # also score substring presence for short queries
            substring_boost = 0.0
            if any(tok and tok in name for tok in q_tokens):
                substring_boost = 0.2
            score = ratio + substring_boost
            if score >= 0.6:  # threshold tuned for short phrases
                best.append((score, opt))

        # sort by score desc and add if not already chosen
        for _score, opt in sorted(best, key=lambda x: x[0], reverse=True):
            if opt not in chosen:
                chosen.append(opt)

    # 3) Common aliases mapping (e.g., "uv" -> UV Water Disinfection)
    aliases = {
        "uv": ["uv", "ultraviolet", "disinfection"],
        "nitrate": ["nitrate", "no3"],
        "e coli": ["e", "coli", "e. coli", "faecal"],
        "core water": ["core", "water", "analysis"],
        "sans 241": ["sans", "241"],
    }
    for key, toks in aliases.items():
        if all(t in q_tokens for t in toks if len(t) > 1) or key in txt:
            for opt in options:
                if key in opt.name.lower() and opt not in chosen:
                    chosen.append(opt)

    return chosen


def _coerce_tests_from_docs(raw_docs) -> List[SelectedTest]:
    """Convert retriever docs into SelectedTest entries (skip address/unpriced)."""
    out: List[SelectedTest] = []
    for d in raw_docs or []:
        meta = getattr(d, "metadata", {}) or {}
        parsed = _parse_doc_content(getattr(d, "page_content", "") or "")

        cat = (str(meta.get("category") or parsed.get("category") or "")).strip().lower()
        if cat in {"address", "drop_off", "drop-off"}:
            continue

        code = str(parsed.get("id") or meta.get("id") or "").strip() or "ITEM"
        name = parsed.get("test_name") or meta.get("test_name") or meta.get("name") or ""
        price_raw = (
            meta.get("price_zar")
            or meta.get("price_ZAR")
            or parsed.get("price_ZAR")
            or parsed.get("price")
        )
        tat_raw = (
            meta.get("tat_days")
            or meta.get("turnaround_days")
            or parsed.get("turnaround_days")
            or parsed.get("tat")
        )

        # price
        price_val = None
        if price_raw not in (None, ""):
            try:
                price_val = float(str(price_raw).replace("R", "").replace(",", "").strip())
            except Exception:
                price_val = None
        if not name or price_val is None or price_val <= 0:
            continue

        # tat
        try:
            tat_days = int(float(tat_raw)) if tat_raw not in (None, "") else 0
        except Exception:
            tat_days = 0

        out.append(SelectedTest(code=code, name=name, price=price_val, tat_days=tat_days))
    return out


def _fmt_option(i: int, t: SelectedTest) -> str:
    # Hide pricing/TAT by default (revealed on request)
    return f"{i}. {t.name}"


def _fmt_line_item(t: SelectedTest) -> str:
    # Hide pricing/TAT by default (revealed on request)
    return f"- {t.name}"


def _total(tests: List[SelectedTest]) -> float:
    return round(sum(t.price for t in tests), 2)


# ---------------- main flow ----------------


def handle_quote_flow(user_text: str, session: ChatSession, retriever) -> str:
    """
    Minimal, robust flow:

        None / ""        -> find options -> ask user to choose
        selection        -> parse selection -> confirm summary -> ask to confirm
        selection_confirm-> if yes: ask for client details; if no: return to options
        confirmation     -> (placeholder for client details -> PDF)
        pdf              -> (placeholder)

    The ChatSession is expected to have:
        - state: str | None
        - last_options: List[SelectedTest]
        - selected_tests: List[SelectedTest]
    """
    q = (user_text or "").strip()

    # If user is in PDF-ready state but asks for new tests, reset flow
    if getattr(session, "state", None) == "pdf":
        s = q.lower()
        # If they explicitly want to generate now, keep guidance
        if ("generate" in s and "pdf" in s) or ("pdf quote" in s):
            return "Say 'generate pdf quote' to produce the document now, or 'back' to continue editing your selection."
        # Detect a new request intent: numbers or test keywords
        if re.search(r"\b\d+\b", s) or any(
            k in s for k in ("water", "soil", "test", "sans", "nitrate", "e. coli", "quote")
        ):
            # Build a short transition summary before resetting
            try:
                if getattr(session, "selected_tests", []):
                    names = ", ".join(t.name for t in session.selected_tests[:4])
                    preface = f"Your previous quote was prepared for: {names}."
                else:
                    preface = "Your previous quote was prepared."
            except Exception:
                preface = "Your previous quote was prepared."
            # Reset to start a fresh selection
            session.state = ""
            session.selected_tests = []
            session.last_options = []
            # Build options next and prepend the transition message
            # fall through to option-building below, using q
            transition_note = preface + " Now let’s look at new options."
        else:
            transition_note = ""
    else:
        transition_note = ""

    # 0) If we have no active flow, produce or reuse options
    st = (getattr(session, "state", None) or "").strip().lower()
    # If we're mid-flow and user asks broadly for new tests, restart selection
    if st in {
        "selection",
        "selection_confirm",
        "confirmation",
        "collect_name",
        "collect_company",
        "collect_email",
        "collect_extra",
    }:
        s = q.lower()
        if any(k in s for k in ("water", "soil", "test", "quote", "sans", "nitrate", "e. coli")):
            session.state = ""
            session.selected_tests = []
            session.last_options = []
    if not st or st == "idle":
        # If user replied with numbers but we have previous options, interpret directly
        if getattr(session, "last_options", None):
            chosen = _match_user_choices(q, session.last_options)
            if chosen:
                existing = {t.code: t for t in getattr(session, "selected_tests", []) or []}
                for t in chosen:
                    existing[t.code] = t
                session.selected_tests = list(existing.values())
                total = _total(session.selected_tests)
                lines = ["Great — here’s your current selection:"]
                for t in session.selected_tests:
                    lines.append(_fmt_line_item(t))
                lines.append(f"\nSubtotal: R{total:,.2f}")
                lines.append(
                    "\nWould you like me to proceed? (yes/no). You can also add more by replying with more numbers."
                )
                session.state = "selection_confirm"
                return "\n".join(lines)

        # Otherwise, fetch fresh options based on the query/topic
        docs = _retrieve_docs(retriever, q or "water test")
        options = _coerce_tests_from_docs(docs)[:5]
        # Retry with broader fallbacks if nothing found
        if not options:
            for alt in (
                "water test",
                "water",
                "soil test",
                "soil",
                "SANS 241",
                "nitrate",
                "e. coli",
            ):
                if alt.lower() in (q.lower() if q else ""):
                    continue
                docs = _retrieve_docs(retriever, alt)
                options = _coerce_tests_from_docs(docs)[:5]
                if options:
                    break
        # Last-resort curated defaults
        if not options:
            qs = (q or "").lower()
            if ("soil" in qs) or ("fertiliser" in qs) or ("fertilizer" in qs):
                curated = [
                    SelectedTest(
                        code="HVB-SOIL-NUT", name="Soil Nutrient Analysis", price=850.0, tat_days=5
                    ),
                    SelectedTest(code="HVB-SOIL-PH", name="Soil pH", price=180.0, tat_days=1),
                    SelectedTest(
                        code="HVB-SOIL-EC",
                        name="Soil Electrical Conductivity",
                        price=220.0,
                        tat_days=1,
                    ),
                    SelectedTest(
                        code="HVB-SOIL-MIC", name="Soil Microbes", price=350.0, tat_days=3
                    ),
                    SelectedTest(
                        code="HVB-SOIL-NIT", name="Soil Nitrate (NO3)", price=260.0, tat_days=2
                    ),
                ]
            else:
                curated = [
                    SelectedTest(
                        code="HVB-CWA", name="Core Water Analysis", price=1700.0, tat_days=5
                    ),
                    SelectedTest(code="HVB-NO3", name="Nitrate", price=441.0, tat_days=2),
                    SelectedTest(
                        code="HVB-ECOLI",
                        name="E. coli or Faecal Coliforms",
                        price=105.0,
                        tat_days=1,
                    ),
                    SelectedTest(code="HVB-TURB", name="Turbidity", price=266.0, tat_days=1),
                    SelectedTest(
                        code="HVB-S241",
                        name="SANS 241 Full Analysis Bundle",
                        price=9600.0,
                        tat_days=6,
                    ),
                ]
            options = curated[:5]

        session.last_options = options
        session.selected_tests = []
        session.state = "selection"

        lines = []
        if transition_note:
            lines.append(transition_note)
        lines.append("I found these options — reply with numbers (e.g., 1 or 1 3):")
        for i, t in enumerate(options, start=1):
            lines.append(_fmt_option(i, t))
        return "\n".join(lines)

    # 1) Selection -> parse user choices
    if session.state == "selection":
        if not q:
            return "Please reply with one or more numbers (e.g., 1 or 1 3)."

        chosen = _match_user_choices(q, getattr(session, "last_options", []) or [])
        if not chosen:
            return "I didn’t catch a valid choice. Please reply with one or more numbers from the list."

        # Merge with any previous picks (dedupe by code)
        existing = {t.code: t for t in getattr(session, "selected_tests", []) or []}
        for t in chosen:
            existing[t.code] = t
        session.selected_tests = list(existing.values())

        # Ask to confirm
        total = _total(session.selected_tests)
        lines = ["Great — here’s your current selection:"]
        for t in session.selected_tests:
            lines.append(_fmt_line_item(t))
        lines.append(f"\nSubtotal: R{total:,.2f}")
        lines.append(
            "\nWould you like me to proceed? (yes/no). You can also add more by replying with more numbers."
        )
        session.state = "selection_confirm"
        return "\n".join(lines)

    # 2) Confirmation
    if session.state == "selection_confirm":
        if _is_negative(q):
            session.state = "selection"
            return "No problem — you can add more items by replying with numbers from the list."

        # Proceed to details ONLY when the user explicitly confirms
        if _is_acknowledgement(q):
            # Before client details, collect the number of samples per test
            session.state = "collect_counts"
            session.collect_counts_index = 0
            session.sample_counts = {}
            if not session.selected_tests:
                session.state = "selection"
                return "Let's reconfirm your tests by replying with numbers."
            t = session.selected_tests[0]
            return f"How many samples would you like tested for '{t.name}'? (Enter a number)"

        # Allow adding more numbers while confirming
        added = _match_user_choices(q, getattr(session, "last_options", []) or [])
        if added:
            existing = {t.code: t for t in getattr(session, "selected_tests", []) or []}
            for t in added:
                existing[t.code] = t
            session.selected_tests = list(existing.values())

            total = _total(session.selected_tests)
            lines = ["Updated selection:"]
            for t in session.selected_tests:
                lines.append(_fmt_line_item(t))
            lines.append(f"\nSubtotal: R{total:,.2f}")
            lines.append("\nProceed? (yes/no)")
            return "\n".join(lines)

        return "Please reply with **yes** to proceed (or **no** to continue choosing). You can also add more numbers."

    # 3) Structured sample count + client details collection
    if session.state == "collect_counts":
        # Expect a positive integer
        m = re.search(r"\b\d+\b", q)
        if not m:
            return "Please enter a number of samples (e.g., 1 or 3)."
        count = max(1, int(m.group(0)))
        idx = int(getattr(session, "collect_counts_index", 0) or 0)
        if idx < 0 or idx >= len(session.selected_tests):
            idx = 0
        cur = session.selected_tests[idx]
        session.sample_counts[cur.code] = count
        idx += 1
        session.collect_counts_index = idx
        if idx < len(session.selected_tests):
            nxt = session.selected_tests[idx]
            return f"And how many samples for '{nxt.name}'?"
        # Done with counts → proceed to client details
        session.state = "collect_name"
        return "Great. First, please provide your name and surname."

    if session.state == "collect_name":
        txt = q.strip()
        if not txt or len(txt.split()) < 2:
            return "Please provide both your name and surname."
        parts = txt.split()
        session.client_name = parts[0]
        session.client_surname = " ".join(parts[1:])
        session.state = "collect_phone"
        return "Thanks. What is the best cellphone number for this quote? (Required)"

    if session.state == "collect_phone":
        txt = q.strip()
        if not re.findall(r"\d", txt) or len(re.findall(r"\d", txt)) < 7:
            return "Please provide a valid phone number (at least 7 digits)."
        session.client_phone = txt
        session.state = "collect_email"
        return "Got it. What is your email address?"

    if session.state == "collect_company":
        txt = q.strip()
        session.client_company = txt
        session.state = "collect_extra"
        return "Would you like to include any extra information or notes on the quote? (Optional)"

    if session.state == "collect_email":
        txt = q.strip()
        if "@" not in txt:
            return "Please provide a valid email address."
        session.client_email = txt
        session.state = "collect_company"
        return "Thanks. What is your company name? (You can say 'personal' if not applicable.)"

    if session.state == "collect_extra":
        session.client_extra_info = q.strip()
        session.state = "pdf"
        summary_lines = ["Thank you — here is a quick summary:"]
        for t in session.selected_tests or []:
            qty = max(1, int((session.sample_counts or {}).get(t.code, 1)))
            summary_lines.append(f"- {t.name} × {qty}")
        summary_lines.append(
            f"Client: {session.client_name} {session.client_surname}, {session.client_company or 'personal'}"
        )
        if session.client_extra_info:
            summary_lines.append(f"Notes: {session.client_extra_info}")
        summary_lines.append(
            "If you'd like, reply 'yes' and I will generate your PDF quote now. You can also ask about pricing/TAT, drop-off, or sample preparation."
        )
        return "\n".join(summary_lines)

    # 4) PDF state (placeholder; actual PDF handled by RAGService.structured_quote)
    if session.state == "pdf":
        return "Say 'generate pdf quote' to produce the document now, or 'back' to continue editing your selection."

    # Fallback safety
    session.state = None
    return "Let’s start a new quote. Which test or sample type are you interested in?"
