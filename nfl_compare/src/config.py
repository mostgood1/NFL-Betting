from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


def _read_first_existing(paths: Iterable[Path]) -> Tuple[bool, str]:
    """Try reading the first existing file from paths. Returns (ok, content)."""
    for p in paths:
        try:
            if p.exists() and p.is_file():
                return True, p.read_text(encoding="utf-8").strip()
        except Exception:
            continue
    return False, ""


def _load_secret_files_fallbacks() -> None:
    """Support secrets mounted as files.

    Conventions handled:
    - <VAR>_FILE environment variable pointing to a file path (e.g., ODDS_API_KEY_FILE)
    - Common secret file mount points for specific variables if <VAR> isn't already set:
      /etc/secrets/<var>, /etc/secrets/<VAR>, /run/secrets/<var>, /run/secrets/<VAR>
    """
    # 1) Handle <VAR>_FILE passthroughs generically
    for k, v in list(os.environ.items()):
        if not k.endswith("_FILE"):
            continue
        base = k[:-5]
        # Don't override if already set
        if os.environ.get(base):
            continue
        try:
            p = Path(v)
            if p.exists() and p.is_file():
                os.environ[base] = p.read_text(encoding="utf-8").strip()
        except Exception:
            # ignore unreadable file
            continue

    # 2) Specific common fallbacks for known keys if still unset
    known_secret_vars: Dict[str, Tuple[str, ...]] = {
        # Odds API key commonly stored as a secret file on hosts like Render
        "ODDS_API_KEY": (
            "/etc/secrets/odds_api_key",
            "/etc/secrets/ODDS_API_KEY",
            "/run/secrets/odds_api_key",
            "/run/secrets/ODDS_API_KEY",
        ),
        # GitHub token for pushes (optional)
        "GITHUB_TOKEN": (
            "/etc/secrets/github_token",
            "/etc/secrets/GITHUB_TOKEN",
            "/run/secrets/github_token",
            "/run/secrets/GITHUB_TOKEN",
        ),
    }
    for var, paths in known_secret_vars.items():
        if os.environ.get(var):
            continue
        ok, content = _read_first_existing(Path(p) for p in paths)
        if ok and content:
            os.environ[var] = content


def load_env() -> None:
    """Load environment from .env files and secret-file fallbacks.

    Searches for .env files:
    - nfl_compare/.env (package root)
    - NFL-Betting/.env (repo root)
    - CWD/.env (current working directory)

    Then loads secret files using <VAR>_FILE and common mount points.
    """
    # Load .env files if python-dotenv is available
    if load_dotenv is not None:
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

    # Always attempt to hydrate env from secret-file conventions
    _load_secret_files_fallbacks()


# Auto-load on import
load_env()
