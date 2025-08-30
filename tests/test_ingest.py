import os
import json
import importlib
from pathlib import Path
import sys

import pytest

from app import config as cfg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FakeEmbeddings:
    def __init__(self, model: str = "", api_key: str | None = None, base_url: str | None = None):
        self.model = model

    def embed_documents(self, texts):
        return [[0.1] * 8 for _ in texts]


class _FakeVectorStore:
    def __init__(self, n_docs: int):
        self.n_docs = n_docs

    def save_local(self, path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "INDEX_SAVED").write_text(str(self.n_docs), encoding="utf-8")


class FakeFAISS:
    @classmethod
    def from_documents(cls, docs, embeddings):
        return _FakeVectorStore(len(docs))


def test_ingest_with_compiled_json(tmp_path, monkeypatch):
    item = {
        "id": "sans-241-full",
        "category": "",
        "test_name": "SANS 241 Full Analysis Bundle",
        "price_ZAR": 9600.0,
        "turnaround_days": 6.0,
        "sample_prep": "",
        "notes": "",
        "embedding_text": "SANS 241 Full Analysis Bundle — Standard: SANS 241 — Price: R9600 — Turnaround: 6 days",
    }
    compiled_dir = tmp_path / "data" / "compiled"
    compiled_dir.mkdir(parents=True, exist_ok=True)
    compiled_json = compiled_dir / "lab_tests_clean.json"
    compiled_json.write_text(json.dumps({"items": [item]}, ensure_ascii=False), encoding="utf-8")

    ingest = importlib.import_module("app.ingest")
    ingest.CFG_DATA_FILE = str(compiled_json)
    ingest.DATA_FILE = str(compiled_json)
    ingest.INDEX_DIR = str(tmp_path / ".vectordb" / "faq_faiss")

    monkeypatch.setattr(ingest, "OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(ingest, "FAISS", FakeFAISS)

    ingest.main()
    assert (Path(ingest.INDEX_DIR) / "INDEX_SAVED").exists()


def test_data_file_exists():
    """Sanity: configured DATA_FILE should exist (or compiled fallback should)."""
    data_file = getattr(cfg, "DATA_FILE", None)
    assert data_file, "CONFIG.DATA_FILE is not set"
    assert os.path.isfile(data_file), f"DATA_FILE not found: {data_file}"


def test_compiled_csv_present():
    """Compiled CSV (fallback) should exist in data/compiled when prepare step ran."""
    compiled = os.path.join(cfg.PROJECT_ROOT, "data", "compiled", "lab_tests_clean.csv")
    # allow tests to pass if compiled file missing but warn
    assert os.path.isfile(compiled), f"Compiled CSV not found: {compiled}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
