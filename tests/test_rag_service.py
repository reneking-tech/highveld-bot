from app.rag_chain import RAGService
import app.config as cfg
import os
import types
import pytest
from app.state import ChatSession

# Ensure OPENAI_API_KEY is set for imports that construct LLM/embeddings.
os.environ.setdefault("OPENAI_API_KEY", "sk-fake-for-tests")


class DummyRetriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, query):
        return self._docs


class DummyDoc:
    """Small test helper document that mimics the simple attributes used by retrievers."""

    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        # Some test suites pass metadata kw; accept both None and dict
        self.metadata = metadata or {}

    def get(self, k, default=None):
        # Provide dict-like access used in some helper code paths
        return self.metadata.get(k, default)


class DummyResp:
    def __init__(self, content: str):
        self.content = content


class DummyLLM:
    """Simple LLM stub that returns predictable content for invoke([...]) calls."""

    def invoke(self, messages):
        # Return a safe short textual response regardless of prompt
        return DummyResp("This is a dummy LLM response used for tests.")

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
    docs = [DummyDoc("id: HVB-0001\ncategory: test\ntest_name: Irrelevant Item\nnotes: none")]
    svc = build_service_with(docs)
    # Force a high threshold to guarantee gating
    monkeypatch.setattr(cfg, "MIN_KEYWORD_OVERLAP", 0.9, raising=False)
    monkeypatch.setattr(cfg, "ENABLE_NOT_SURE_FALLBACK", True, raising=False)

    result = svc.answer_question("What is the price of the SANS 241 full test?")
    assert "not completely sure" in result["answer"].lower()
    assert result["trace"].get("gated") is True


def test_extra_context_enhances_overlap(monkeypatch):
    docs = [DummyDoc("id: HVB-0002\ncategory: test\ntest_name: SANS 241 full\nprice_ZAR: 9200")]
    svc = build_service_with(docs)
    # Set a moderate threshold
    monkeypatch.setattr(cfg, "MIN_KEYWORD_OVERLAP", 0.3, raising=False)
    monkeypatch.setattr(cfg, "ENABLE_NOT_SURE_FALLBACK", True, raising=False)

    # With helpful extra context, we should avoid gating
    extra = "Includes SANS 241 full test and pricing context"
    result = svc.answer_question("How much is a full SANS 241 test?", extra_context=extra)
    assert result["trace"].get("gated") in (False, None)


def test_price_guard_blocks_invented_prices(monkeypatch):
    # Context with no price info
    docs = [
        DummyDoc("id: HVB-0100\ncategory: test\ntest_name: SANS 241 full\nnotes: includes metals")
    ]
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


class DummyMem:
    def __init__(self):
        self.saved = []

    def save_turn(self, u, a):
        self.saved.append((u, a))

    def load_summary(self):
        return "Earlier, user asked about soil tests."


def test_memory_is_saved_and_loaded(monkeypatch):
    # Fake retriever to avoid FAISS
    svc = RAGService(llm_model="gpt-4o-mini", temperature=0.0, top_k=1, memory=DummyMem())
    # Monkeypatch retriever to return empty context
    svc.retriever = types.SimpleNamespace(invoke=lambda q: [])
    # Monkeypatch model to echo

    class FakeResp:
        content = "Here are the options."

    svc.llm = types.SimpleNamespace(invoke=lambda msgs: FakeResp())
    out = svc.answer_question("how much is soil test?")
    assert "answer" in out and isinstance(out["answer"], str)
    assert svc.mem.load_summary()


@pytest.fixture
def rag():
    svc = RAGService()
    # swap in deterministic LLMs
    svc.llm = DummyLLM()
    svc.conversation_llm = DummyLLM()
    # inject retriever returning deterministic docs
    docs = [
        DummyDoc(
            "test_name: Nitrate\nprice_ZAR: 441\nturnaround_days: 2",
            metadata={"price_zar": 441, "test_name": "Nitrate", "turnaround_days": 2},
        ),
        DummyDoc(
            "test_name: Core Water Analysis\nprice_ZAR: 1700\nturnaround_days: 5",
            metadata={"price_zar": 1700, "test_name": "Core Water Analysis", "turnaround_days": 5},
        ),
    ]
    svc.retriever = DummyRetriever(docs)
    return svc


def test_greeting_returns_text(rag):
    resp = rag.answer_question("Hello")
    assert isinstance(resp, dict)
    assert "answer" in resp
    assert isinstance(resp["answer"], str)
    assert resp["answer"].strip() != ""


def test_get_quote_flow_starts_selection(rag):
    session = ChatSession()
    resp = rag.answer_question("Get a quote", session=session)
    assert isinstance(resp, dict)
    assert isinstance(resp["answer"], str)
    # either a selection menu or a helpful quote prompt should be returned
    assert resp["answer"].strip() != ""


def test_structured_quote_returns_json(rag):
    obj = rag.structured_quote("Quote for nitrate")
    assert isinstance(obj, dict)
    assert "tests" in obj
    assert isinstance(obj["tests"], list)
    assert obj["total_price_ZAR"] == pytest.approx(441.0)
