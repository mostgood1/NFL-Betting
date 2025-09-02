from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


def load_env() -> None:
    """Load environment from one or more local .env files if python-dotenv is available.
    Searches:
    - nfl_compare/.env (package root)
    - NFL-Betting/.env (repo root)
    - CWD/.env (current working directory)
    """
    if load_dotenv is None:
        return
    pkg_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [pkg_root / ".env", repo_root / ".env", Path.cwd() / ".env"]
    for fp in candidates:
        try:
            if fp.exists():
                # Do not override already-set vars by default
                load_dotenv(dotenv_path=str(fp), override=False)
        except Exception:
            continue


# Auto-load on import
load_env()
