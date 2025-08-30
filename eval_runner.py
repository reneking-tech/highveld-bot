from __future__ import annotations
from app.rag_chain import RAGService
from app.state import ChatSession
import os
import json

# Ensure project root is on sys.path before importing app modules
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_training_examples(json_path: str) -> list[dict]:
    """Load training examples from JSON file."""
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Using default tests.")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    examples = []
    for category in data.get("categories", []):
        for example in category.get("examples", []):
            examples.append(
                {
                    "question": example.get("question", ""),
                    "expected_answer": example.get("answer", ""),
                    "intent": example.get("intent", ""),
                    "category": category.get("name", ""),
                }
            )
    return examples


def run_once(prompt: str) -> str:
    """Call RAGService once with a fresh session and return the answer text."""
    try:
        svc = RAGService()
        sess = ChatSession()
        res = svc.answer_question(prompt, session=sess)
        return res.get("answer", "No answer returned")
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    # Path to training examples
    training_json = os.path.join(PROJECT_ROOT, "data", "training_examples_full.json")

    # Load examples
    examples = load_training_examples(training_json)
    if not examples:
        # Fallback to default tests if JSON not found
        examples = [
            {
                "question": "Hi there",
                "expected_answer": "Greeting response",
                "intent": "greeting",
                "category": "Default",
            },
            {
                "question": "I need a water test quote",
                "expected_answer": "Quote response",
                "intent": "quote",
                "category": "Default",
            },
            {
                "question": "Add Soil Structure as well",
                "expected_answer": "Add to quote",
                "intent": "quote",
                "category": "Default",
            },
            {
                "question": "Yes, generate the PDF",
                "expected_answer": "PDF generation",
                "intent": "quote",
                "category": "Default",
            },
            {
                "question": "John",
                "expected_answer": "Name collection",
                "intent": "quote",
                "category": "Default",
            },
            {
                "question": "Doe",
                "expected_answer": "Surname collection",
                "intent": "quote",
                "category": "Default",
            },
            {
                "question": "Acme Pty Ltd",
                "expected_answer": "Company collection",
                "intent": "quote",
                "category": "Default",
            },
            {
                "question": "john.doe@example.com",
                "expected_answer": "Email collection",
                "intent": "quote",
                "category": "Default",
            },
        ]

    print(f"Loaded {len(examples)} training examples from {training_json}")
    print("=" * 80)

    for i, ex in enumerate(examples, 1):
        question = ex["question"]
        expected = ex["expected_answer"]
        intent = ex.get("intent", "unknown")
        category = ex.get("category", "unknown")

        print(f"Test {i}/{len(examples)}")
        print(f"Category: {category} | Intent: {intent}")
        print(f"Q: {question}")
        print(f"Expected: {expected}")

        actual = run_once(question)
        print(f"Actual: {actual}")
        print("-" * 80)


if __name__ == "__main__":
    main()
