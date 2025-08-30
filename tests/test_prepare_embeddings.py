from pathlib import Path
import sys
import os
import pytest
from app import config as cfg
from unittest.mock import patch
from app.prepare_embeddings import prepare_data

# Ensure project root on sys.path when running directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def write_csv(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")


def test_prepare_embeddings_basic():
    # The pipeline should produce a compiled CSV/JSON under data/compiled
    compiled_csv = os.path.join(
        cfg.PROJECT_ROOT, "data", "compiled", "lab_tests_clean.csv")
    compiled_json = os.path.join(
        cfg.PROJECT_ROOT, "data", "compiled", "lab_tests_clean.json")
    assert os.path.isfile(compiled_csv) or os.path.isfile(
        compiled_json
    ), "Compiled dataset missing in data/compiled"


def test_prepare_data_success():
    """Test that prepare_data runs without errors."""
    with patch("app.prepare_embeddings.ingest_main") as mock_ingest:
        mock_ingest.return_value = None
        prepare_data()
        mock_ingest.assert_called_once()


def test_prepare_data_failure():
    """Test that prepare_data raises an error on failure."""
    with patch("app.prepare_embeddings.ingest_main") as mock_ingest:
        mock_ingest.side_effect = Exception("Ingestion failed")
        with pytest.raises(Exception, match="Ingestion failed"):
            prepare_data()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
