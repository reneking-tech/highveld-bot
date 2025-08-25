import app.config as cfg
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import re
import warnings
import sys
from dotenv import load_dotenv, find_dotenv
import io
import csv
import pandas as pd

# Ensure project root is on sys.path when run as a script (python app/ingest.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure env vars are loaded (safety net for OPENAI_API_KEY)
load_dotenv(find_dotenv())

DATA_FILE = getattr(cfg, "DATA_FILE", os.path.join(
    PROJECT_ROOT, "data", "lab_faq.csv"))
INDEX_DIR = getattr(cfg, "INDEX_DIR", os.path.join(
    PROJECT_ROOT, ".vectordb", "faq_faiss"))
CHUNK_SIZE = int(getattr(cfg, "CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(getattr(cfg, "CHUNK_OVERLAP", 100))
EMBEDDING_MODEL = str(
    getattr(cfg, "EMBEDDING_MODEL", "text-embedding-3-large"))

warnings.filterwarnings(
    "ignore",
    message="`pydantic.error_wrappers:ValidationError` has been moved"
)


def main() -> None:
    # 0) Basic checks
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Expected CSV at: {DATA_FILE}")

    # 1) Load CSV; tolerate mixed-schema files by splitting on a secondary header
    def _load_mixed_csv(path: str) -> pd.DataFrame:
        records: list[dict] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=",", quotechar='"')
            for raw in reader:
                # Skip completely empty rows
                if not raw or all((str(c).strip() == "" for c in raw)):
                    continue
                lower = [str(c).strip().lower() for c in raw]
                # Skip any header rows (we may have more than one)
                if lower[:5] == [
                    "test_name",
                    "price_zar",
                    "turnaround_days",
                    "sample_prep",
                    "notes",
                ]:
                    continue
                if lower[:7] == [
                    "id",
                    "category",
                    "test_name",
                    "price_zar",
                    "turnaround_days",
                    "sample_prep",
                    "notes",
                ]:
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

    df = _load_mixed_csv(DATA_FILE)
    if df.empty:
        raise ValueError("CSV is empty or unreadable; please add rows before ingesting.")

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

    documents: list[Document] = []
    for idx, row in df.iterrows():
        rownum = int(idx) + 1
        name = str(row.get("test_name", "")).strip()
        # Accept both price_ZAR and price_zar casings
        price = row.get("price_ZAR", row.get("price_zar", ""))
        tat = row.get("turnaround_days", "")
        sample_prep = str(row.get("sample_prep", "")).strip()
        notes = str(row.get("notes", "")).strip()

        # Prefer provided id/category when present
        provided_id = str(row.get("id", "")).strip()
        hvb_id = provided_id if provided_id else _extract_hvb_id(name, notes, rownum)
        provided_cat = str(row.get("category", "")).strip().lower()
        is_drop = "drop-off" in name.lower() or "drop off" in name.lower() or provided_cat in {"address", "drop_off", "drop-off"}
        category = provided_cat if provided_cat else ("drop_off" if is_drop else "test")
        address = _extract_address(notes)

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

    # 2) Split to chunks (most rows are already short; this is just to be safe)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No documents to index after normalization.")

    # 3) Embed + build vector store
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = FAISS.from_documents(chunks, embeddings)

    # 4) Persist locally
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectordb.save_local(INDEX_DIR)

    print(f"✅ Index built and saved: {INDEX_DIR}")
    print(f"   Documents: {len(documents)}  |  Chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
