try:
    import app.config as cfg
except ModuleNotFoundError:
    import config as cfg  # fallback when run as script

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import re
import warnings
import sys
from dotenv import load_dotenv, find_dotenv
import csv
import pandas as pd
import json

# Ensure project root is on sys.path when run as a script (python app/ingest.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure env vars are loaded (safety net for OPENAI_API_KEY)
load_dotenv(find_dotenv())
# Also try loading .env from the repo root explicitly (more robust in subprocesses)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

# Prefer compiled JSON; fall back to combined/compiled CSV/faq CSV if not found (unless overridden in config)
_DEFAULT_PREFERRED = os.path.join(PROJECT_ROOT, "data", "compiled", "lab_tests_clean.json")
_FALLBACKS = [
    os.path.join(PROJECT_ROOT, "data", "compiled", "lab_tests_clean.csv"),
    os.path.join(PROJECT_ROOT, "data", "lab_tests_combined.csv"),
    os.path.join(PROJECT_ROOT, "data", "lab_faq.csv"),
]
CFG_DATA_FILE = getattr(cfg, "DATA_FILE", "").strip() if hasattr(cfg, "DATA_FILE") else ""
DATA_FILE = CFG_DATA_FILE or _DEFAULT_PREFERRED

INDEX_DIR = getattr(cfg, "INDEX_DIR", os.path.join(PROJECT_ROOT, ".vectordb", "faq_faiss"))
CHUNK_SIZE = int(getattr(cfg, "CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(getattr(cfg, "CHUNK_OVERLAP", 100))
EMBEDDING_MODEL = str(getattr(cfg, "EMBEDDING_MODEL", "text-embedding-3-large"))

warnings.filterwarnings(
    "ignore", message="`pydantic.error_wrappers:ValidationError` has been moved"
)


def main() -> None:
    # Resolve input path with graceful fallbacks unless user explicitly set cfg.DATA_FILE
    data_path = DATA_FILE
    if not os.path.exists(data_path):
        if CFG_DATA_FILE:
            raise FileNotFoundError(f"Expected DATA_FILE from config not found: {data_path}")
        for fp in [_DEFAULT_PREFERRED, *_FALLBACKS]:
            if os.path.exists(fp):
                data_path = fp
                break
        else:
            raise FileNotFoundError(
                f"No data file found. Tried: {_DEFAULT_PREFERRED}, {_FALLBACKS}"
            )

    # Load JSON (compiled) or CSV (mixed)
    def _is_header_row(lower_row: list[str]) -> bool:
        # normalize typical headers regardless of casing/order extensions
        # 1) legacy minimal schema
        if lower_row[:5] == ["test_name", "price_zar", "turnaround_days", "sample_prep", "notes"]:
            return True
        # 2) compiled/clean schema (CSV) possibly with extra columns like embedding_text
        if (
            len(lower_row) >= 7
            and lower_row[0] == "id"
            and lower_row[1] == "category"
            and lower_row[2] == "test_name"
        ):
            return True
        # 3) combined schema starting with category,test_name
        if len(lower_row) >= 2 and lower_row[0] == "category" and lower_row[1] == "test_name":
            return True
        return False

    def _load_mixed_csv(path: str) -> pd.DataFrame:
        records: list[dict] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=",", quotechar='"')
            for raw in reader:
                if not raw or all((str(c).strip() == "" for c in raw)):
                    continue
                lower = [str(c).strip().lower() for c in raw]
                # Skip any header rows (supports multiple header shapes)
                if _is_header_row(lower):
                    continue

                # Normalize into the superset schema
                if len(raw) <= 5:
                    # test_name, price_ZAR, turnaround_days, sample_prep, notes
                    row = [c.strip() for c in raw] + [""] * (5 - len(raw))
                    rec = {
                        "id": "",
                        "category": "",
                        "test_name": row[0],
                        "price_ZAR": row[1],
                        "turnaround_days": row[2],
                        "sample_prep": row[3],
                        "notes": row[4],
                    }
                    records.append(rec)
                else:
                    # id, category, test_name, price_ZAR, turnaround_days, sample_prep, notes(+ extras)
                    row = [c.strip() for c in raw]
                    # Pad to at least 7
                    if len(row) < 7:
                        row += [""] * (7 - len(row))
                    head = row[:6]
                    tail = row[6:]
                    notes = ",".join(tail).strip()
                    rec = {
                        "id": head[0],
                        "category": head[1],
                        "test_name": head[2],
                        "price_ZAR": head[3],
                        "turnaround_days": head[4],
                        "sample_prep": head[5],
                        "notes": notes,
                    }
                    records.append(rec)

        return pd.DataFrame.from_records(records)

    def _load_compiled_json(path: str) -> pd.DataFrame:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError(f"Unrecognized JSON format in {path}")
        # Keep common fields; embedding_text if present will be used as page_content
        rows = []
        for r in items:
            if not isinstance(r, dict):
                continue
            rows.append(
                {
                    "id": str(r.get("id", "") or "").strip(),
                    "category": str(r.get("category", "") or "").strip(),
                    "test_name": str(r.get("test_name", "") or "").strip(),
                    "price_ZAR": r.get("price_ZAR", ""),
                    "turnaround_days": r.get("turnaround_days", ""),
                    "sample_prep": str(r.get("sample_prep", "") or "").strip(),
                    "notes": str(r.get("notes", "") or "").strip(),
                    "embedding_text": str(r.get("embedding_text", "") or "").strip(),
                }
            )
        return pd.DataFrame.from_records(rows)

    if data_path.lower().endswith(".json"):
        df = _load_compiled_json(data_path)
    else:
        df = _load_mixed_csv(data_path)

    if df.empty:
        raise ValueError("CSV/JSON is empty or unreadable; please add rows before ingesting.")

    def _extract_hvb_id(name: str, notes: str, rownum: int) -> str:
        text = f"{name}\n{notes}"
        m = re.search(r"\bHVB-(\d{4})\b", text)
        if m:
            return f"HVB-{m.group(1)}"
        return f"HVB-{rownum:04d}"

    def _extract_address(notes: str) -> str:
        if not isinstance(notes, str):
            return ""
        m = re.search(r"(?i)(physical\s+address|address)\s*:\s*(.+)$", notes)
        if m:
            return m.group(2).strip()
        return ""

    def _looks_bad(name: str, category: str) -> bool:
        # Skip rows that are clearly numeric-only or empty
        def _numy(s: str) -> bool:
            s = str(s or "").strip()
            try:
                float(s.replace(",", ""))
                return True
            except Exception:
                return False

        has_alpha = bool(re.search(r"[A-Za-z]", str(name or "")))
        return (not has_alpha) or (_numy(name) and _numy(category))

    documents: list[Document] = []
    for idx, row in df.iterrows():
        rownum = int(idx) + 1
        name = str(row.get("test_name", "")).strip()
        if _looks_bad(name, str(row.get("category", "")).strip()):
            continue
        price = row.get("price_ZAR", row.get("price_zar", ""))
        tat = row.get("turnaround_days", "")
        sample_prep = str(row.get("sample_prep", "")).strip()
        notes = str(row.get("notes", "")).strip()
        embedding_text = (
            str(row.get("embedding_text", "")).strip() if "embedding_text" in row else ""
        )

        # Prefer provided id/category when present
        provided_id = str(row.get("id", "")).strip()
        hvb_id = provided_id if provided_id else _extract_hvb_id(name, notes, rownum)
        provided_cat = str(row.get("category", "")).strip().lower()
        is_drop = (
            "drop-off" in name.lower()
            or "drop off" in name.lower()
            or provided_cat in {"address", "drop_off", "drop-off"}
        )
        category = provided_cat if provided_cat else ("drop_off" if is_drop else "test")
        address = _extract_address(notes)

        if embedding_text:
            page = embedding_text
        else:
            lines = [
                f"id: {hvb_id}",
                f"category: {category}",
            ]
            if name:
                lines.append(f"test_name: {name}")
            if price != "" and not pd.isna(price):
                lines.append(f"price_ZAR: {price}")
            if tat != "" and not pd.isna(tat):
                lines.append(f"turnaround_days: {tat}")
            if address:
                lines.append(f"address: {address}")
            if sample_prep:
                lines.append(f"sample_prep: {sample_prep}")
            if notes:
                lines.append(f"notes: {notes}")
            page = "\n".join(lines)

        documents.append(
            Document(
                page_content=page,
                metadata={"row": rownum, "id": hvb_id, "category": category},
            )
        )

    # 2) Split to chunks only if needed (most rows are short)
    if not documents:
        raise ValueError("No documents to index after normalization.")
    max_len = max(len(d.page_content) for d in documents)
    if max_len <= CHUNK_SIZE:
        chunks = documents
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise ValueError("No documents to index after normalization.")

    # 3) Embed + build vector store
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Ensure it is present in .env or environment.")
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    if base_url:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key, base_url=base_url)
    else:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)

    vectordb = FAISS.from_documents(chunks, embeddings)

    # 4) Persist locally
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectordb.save_local(INDEX_DIR)

    print(f"✅ Index built and saved: {INDEX_DIR}")
    print(f"   Source: {os.path.relpath(data_path, PROJECT_ROOT)}")
    print(f"   Documents: {len(documents)}  |  Chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
