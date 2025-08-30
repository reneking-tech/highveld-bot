"""
Prepare embeddings by normalizing and compiling data into JSON/CSV format.
Provides a thin wrapper (`ingest_main`) around `app.ingest.main` so tests can mock it.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure app imports resolve when run as a script
sys.path.insert(0, str(Path(__file__).parent))


def ingest_main() -> None:
    """Delegate to the real ingestion entrypoint."""
    from app.ingest import main as _main

    _main()


def prepare_data() -> None:
    """Prepare and compile lab test data for embeddings (invokes ingest)."""
    try:
        ingest_main()
    except Exception:
        # Bubble up so tests can assert
        raise


if __name__ == "__main__":
    prepare_data()
