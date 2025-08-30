#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv  # added

BASE_DIR = Path(__file__).resolve().parents[1]
# load .env so OPENAI_API_KEY is available to subprocesses
load_dotenv(find_dotenv())


def _sanitize_openai_base_env(env: dict) -> None:
    """Ensure base URL is valid or unset so SDK defaults to api.openai.com/v1."""
    val = (env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE") or "").strip()
    if not val:
        env.pop("OPENAI_BASE_URL", None)
        env.pop("OPENAI_API_BASE", None)
        return
    if not (val.startswith("http://") or val.startswith("https://")):
        val = "https://" + val
    env["OPENAI_BASE_URL"] = val
    env["OPENAI_API_BASE"] = val


def run(cmd, env=None):
    print("> " + " ".join(cmd))
    # apply env sanitisation on each call
    env = env or os.environ.copy()
    _sanitize_openai_base_env(env)
    subprocess.check_call(cmd, cwd=BASE_DIR, env=env)


def main():
    py = sys.executable
    # Optional early sanity check (friendlier than ingest's crash)
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it in your shell.")
        sys.exit(1)
    # Normalize CSV -> compiled JSON/CSV (one source of truth)
    run([py, "-m", "app.prepare_embeddings"])
    # Build FAISS index used by the app
    run([py, "-m", "app.ingest"])


if __name__ == "__main__":
    main()
