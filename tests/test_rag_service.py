import app.config as cfg
from app.rag_chain import RAGService, _keyword_overlap_ratio
import os
import types
import pytest

# Ensure OPENAI_API_KEY is set for imports that construct LLM/embeddings.
os.environ.setdefault("OPENAI_API_KEY", "sk-fake-for-tests")


class DummyRetriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, query):
        return self._docs


class DummyDoc:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}


class DummyLLM:
    def __init__(self):
        self.model = "dummy"

    def invoke(self, messages):
        class R:
            content = "OK"
        return R()

    def stream(self, messages):
        yield types.SimpleNamespace(content="OK")


def build_service_with(docs):
    svc = RAGService(llm_model="gpt-4o-mini")
    # Monkeypatch retriever and llm
    svc.retriever = DummyRetriever(docs)
    svc.llm = DummyLLM()
    return svc


def test_address_override_returns_config_address(monkeypatch):
    svc = build_service_with([])
    ans = svc.answer_question("What is your address for drop off?")
    assert isinstance(ans, dict)
    assert cfg.DROP_OFF_ADDRESS.split(" ")[0] in ans["answer"]
    assert ans["sources"] == []
    assert ans["trace"].get("override") == "DROP_OFF_ADDRESS"


def test_context_gating_triggers_not_sure(monkeypatch):
    # Very weak context relative to the question
    docs = [
        DummyDoc(
            "id: HVB-0001\ncategory: test\ntest_name: Irrelevant Item\nnotes: none")
    ]
    svc = build_service_with(docs)
    # Force a high threshold to guarantee gating
    monkeypatch.setattr(cfg, "MIN_KEYWORD_OVERLAP", 0.9, raising=False)
    monkeypatch.setattr(cfg, "ENABLE_NOT_SURE_FALLBACK", True, raising=False)

    result = svc.answer_question(
        "What is the price of the SANS 241 full test?")
    assert "not completely sure" in result["answer"].lower()
    assert result["trace"].get("gated") is True


def test_extra_context_enhances_overlap(monkeypatch):
    docs = [DummyDoc(
        "id: HVB-0002\ncategory: test\ntest_name: SANS 241 full\nprice_ZAR: 9200")]
    svc = build_service_with(docs)
    # Set a moderate threshold
    monkeypatch.setattr(cfg, "MIN_KEYWORD_OVERLAP", 0.3, raising=False)
    monkeypatch.setattr(cfg, "ENABLE_NOT_SURE_FALLBACK", True, raising=False)

    # With helpful extra context, we should avoid gating
    extra = "Includes SANS 241 full test and pricing context"
    result = svc.answer_question(
        "How much is a full SANS 241 test?", extra_context=extra)
    assert result["trace"].get("gated") in (False, None)


def test_price_guard_blocks_invented_prices(monkeypatch):
    # Context with no price info
    docs = [DummyDoc("id: HVB-0100\ncategory: test\ntest_name: SANS 241 full\nnotes: includes metals")]
    svc = build_service_with(docs)
    # Force the LLM to emit a price-like string
    class PriceyLLM(DummyLLM):
        def invoke(self, messages):
            class R:
                content = "The price is R9999.00"
            return R()
    svc.llm = PriceyLLM()
    monkeypatch.setattr(cfg, "ENABLE_PRICE_GUARD", True, raising=False)

    out = svc.answer_question("What is the price for SANS 241 full test?")
    assert "not completely sure" in out["answer"].lower()
