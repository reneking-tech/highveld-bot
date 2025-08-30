"""
Tool for testing the bot against training examples.
This script allows testing the RAG chain with examples from the training data.
"""

from app.state import ChatSession
from app.rag_chain import RAGService
import os
import json
import sys
import random
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))


def load_examples(category=None, intent=None, count=None, filename="training_examples_full.json"):
    """
    Load examples from the training JSON file, optionally filtered by category and/or intent.

    Args:
        category: Optional category name to filter by
        intent: Optional intent to filter by
        count: Optional limit on number of examples to return
        filename: JSON file to load from (default: training_examples_full.json)

    Returns:
        List of example dictionaries
    """
    training_file = os.path.join(parent_dir, "data", filename)

    if not os.path.exists(training_file):
        # Try the default file if specified file doesn't exist
        if filename != "training_examples.json":
            print(f"Training file not found: {training_file}, trying training_examples.json")
            return load_examples(category, intent, count, "training_examples.json")
        else:
            print(f"No training file found: {training_file}")
            return []

    with open(training_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for cat in data.get("categories", []):
        if category and cat["name"].lower() != category.lower():
            continue

        for ex in cat.get("examples", []):
            if intent and ex.get("intent", "").lower() != intent.lower():
                continue

            examples.append(
                {
                    "category": cat["name"],
                    "intent": ex.get("intent", ""),
                    "question": ex.get("question", ""),
                    "answer": ex.get("answer", ""),
                }
            )

        if category and not intent:
            # If specific category requested and no intent filter, break after first match
            break

    # Apply count limit if specified
    if count and count > 0:
        examples = examples[:count]

    return examples


def list_categories(filename="training_examples_full.json"):
    """List all available categories in the training data."""
    training_file = os.path.join(parent_dir, "data", filename)

    if not os.path.exists(training_file):
        if filename != "training_examples.json":
            return list_categories("training_examples.json")
        return []

    with open(training_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [cat["name"] for cat in data.get("categories", [])]


def list_intents(filename="training_examples_full.json"):
    """List all available intents in the training data."""
    training_file = os.path.join(parent_dir, "data", filename)

    if not os.path.exists(training_file):
        if filename != "training_examples.json":
            return list_intents("training_examples.json")
        return []

    with open(training_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    intents = set()
    for cat in data.get("categories", []):
        for ex in cat.get("examples", []):
            if "intent" in ex and ex["intent"]:
                intents.add(ex["intent"])

    return sorted(list(intents))


def test_example(service, example, session=None):
    """Test a single example against the RAG service."""
    question = example["question"]
    expected = example["answer"]
    category = example["category"]
    intent = example["intent"]

    print(f"\n[CATEGORY: {category} | INTENT: {intent}]")
    print(f"QUESTION: {question}")
    print(f"EXPECTED: {expected}")

    # Use the service to get an answer
    result = service.answer_question(question, session=session)
    answer = result.get("answer", "")

    print(f"ACTUAL: {answer}")
    print("-" * 80)

    return {
        "question": question,
        "expected": expected,
        "actual": answer,
        "category": category,
        "intent": intent,
    }


def main():
    """Main function to run tests."""
    # Initialize the RAG service
    service = RAGService()
    session = ChatSession()

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Test the chatbot with training examples")
    parser.add_argument("--category", "-c", help="Filter by category")
    parser.add_argument("--intent", "-i", help="Filter by intent")
    parser.add_argument("--random", "-r", type=int, help="Test N random examples")
    parser.add_argument("--all", "-a", action="store_true", help="Test all examples")
    parser.add_argument(
        "--list-categories", "-lc", action="store_true", help="List available categories"
    )
    parser.add_argument("--list-intents", "-li", action="store_true", help="List available intents")
    parser.add_argument(
        "--file", "-f", default="training_examples_full.json", help="Training file to use"
    )
    args = parser.parse_args()

    # List categories or intents if requested
    if args.list_categories:
        categories = list_categories(args.file)
        print("Available categories:")
        for cat in categories:
            print(f"- {cat}")
        return

    if args.list_intents:
        intents = list_intents(args.file)
        print("Available intents:")
        for intent in intents:
            print(f"- {intent}")
        return

    # Load examples
    examples = load_examples(args.category, args.intent, filename=args.file)

    if not examples:
        print("No matching examples found.")
        return

    print(f"Found {len(examples)} matching examples.")

    # Select examples to test
    if args.random and args.random > 0:
        if args.random > len(examples):
            print(f"Only {len(examples)} examples available, testing all.")
        else:
            examples = random.sample(examples, min(args.random, len(examples)))
    elif not args.all:
        # Default: test first example only
        examples = examples[:1]

    # Run tests
    results = []
    for example in examples:
        results.append(test_example(service, example, session))

    print(f"\nTested {len(results)} examples.")


if __name__ == "__main__":
    main()
