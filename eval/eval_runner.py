"""
Run training examples and rate answers (Jaccard + sequence ratio).

Usage:
  python eval/eval_runner.py            # runs all examples (fresh session per example)
  python eval/eval_runner.py --limit 50
  python eval/eval_runner.py --out eval/results.jsonl --pause 0.2
"""

from __future__ import annotations
from app.state import ChatSession
from app.rag_chain import RAGService
from typing import List, Dict, Any
import difflib
import re
import argparse
import time
import json
import os

# Ensure project root is on sys.path BEFORE importing app modules
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --- simple text similarity helpers --- #


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()


def _token_set(s: str) -> set:
    return set(t for t in _normalize_text(s).split() if t)


def _jaccard(a: str, b: str) -> float:
    A = _token_set(a)
    B = _token_set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _seq_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def _combined_score(expected: str, actual: str) -> float:
    j = _jaccard(expected, actual)
    s = _seq_ratio(expected, actual)
    return (j + s) / 2.0


# --- load examples --- #


def load_examples(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for cat in data.get("categories", []):
        for ex in cat.get("examples", []):
            q = ex.get("question", "") or ""
            expected = ex.get("answer", "") or ""
            if not q:
                continue
            out.append({"category": cat.get("name", ""), "question": q, "expected": expected})
    return out


# --- runner --- #


def run_examples(
    examples: List[Dict[str, Any]],
    limit: int = 0,
    pause: float = 0.2,
    out_path: str = None,
    conversation_mode: bool = False,
) -> Dict[str, Any]:
    svc = RAGService()
    results = []
    total = 0
    passed_threshold = 0
    threshold = 0.55
    session = ChatSession() if conversation_mode else None

    for i, ex in enumerate(examples):
        if limit and total >= limit:
            break
        q = ex["question"]
        expected = ex.get("expected", "") or ""
        # reuse session when conversation_mode true, else fresh session per example
        sess = session if conversation_mode else ChatSession()
        try:
            res = svc.answer_question(q, session=sess)
            actual = res.get("answer", "") or ""
        except Exception as e:
            actual = f"ERROR: {e}"
        score = _combined_score(expected, actual)
        j = _jaccard(expected, actual)
        s = _seq_ratio(expected, actual)
        record = {
            "index": i + 1,
            "category": ex.get("category", ""),
            "question": q,
            "expected": expected,
            "actual": actual,
            "jaccard": round(j, 4),
            "seq_ratio": round(s, 4),
            "score": round(score, 4),
        }
        results.append(record)
        total += 1
        if score >= threshold:
            passed_threshold += 1
        if out_path:
            with open(out_path, "a", encoding="utf-8") as fo:
                fo.write(json.dumps(record, ensure_ascii=False) + "\n")
        time.sleep(pause)

    summary = {
        "examples_tested": total,
        "avg_score": round(sum(r["score"] for r in results) / total, 4) if total else 0.0,
        "passed_threshold": passed_threshold,
        "pass_rate": round(passed_threshold / total, 4) if total else 0.0,
        "threshold": threshold,
    }
    return {"summary": summary, "results": results}


# --- CLI --- #


def main():
    parser = argparse.ArgumentParser(description="Eval runner for training examples")
    parser.add_argument(
        "--examples",
        "-e",
        default=os.path.join(PROJECT_ROOT, "data", "training_examples_full.json"),
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=0, help="Limit number of examples (0 = all)"
    )
    parser.add_argument("--out", "-o", default=os.path.join(PROJECT_ROOT, "eval", "results.jsonl"))
    parser.add_argument("--pause", type=float, default=0.2, help="Seconds to pause between queries")
    args = parser.parse_args()

    examples = load_examples(args.examples)
    conversation_mode = False
    if not examples:
        # fallback small conversational sequence (preserve session across steps)
        examples = [
            {"question": "Hi there", "expected": "Greeting response", "category": "Default"},
            {
                "question": "I need a water test quote",
                "expected": "Quote response",
                "category": "Default",
            },
            {
                "question": "Add Soil Structure as well",
                "expected": "Add to quote",
                "category": "Default",
            },
            {
                "question": "Yes, generate the PDF",
                "expected": "PDF generation",
                "category": "Default",
            },
            {"question": "John", "expected": "Name collection", "category": "Default"},
            {"question": "Doe", "expected": "Surname collection", "category": "Default"},
            {"question": "Acme Pty Ltd", "expected": "Company collection", "category": "Default"},
            {
                "question": "john.doe@example.com",
                "expected": "Email collection",
                "category": "Default",
            },
        ]
        conversation_mode = True

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)

    print(f"Loaded {len(examples)} training examples from {args.examples}")
    print("=" * 80)
    res = run_examples(
        examples,
        limit=args.limit,
        pause=args.pause,
        out_path=args.out,
        conversation_mode=conversation_mode,
    )
    summary = res["summary"]
    print("=== Summary ===")
    print(f"Examples tested: {summary['examples_tested']}")
    print(f"Average score: {summary['avg_score']:.4f}")
    print(
        f"Pass threshold: {summary['threshold']}, Passed: {summary['passed_threshold']} ({summary['pass_rate']*100:.1f}%)"
    )
    print(f"Per-example results written to: {args.out}")
    # save aggregated summary
    agg_path = os.path.join(os.path.dirname(args.out), "summary.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved aggregate summary to {agg_path}")


if __name__ == "__main__":
    main()
