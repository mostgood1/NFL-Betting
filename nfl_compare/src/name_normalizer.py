"""
Utilities for normalizing player names consistently across feeds.

Provides three keys used throughout reconciliation:
- normalize_name: lowercased, trimmed, single-spaced (strict)
- normalize_name_loose: alphanumeric-only, lowercased (loose)
- normalize_alias_init_last: first-initial + last-name, lower, alnum

Enhancements over older ad-hoc versions:
- Strips common suffixes (Jr, Sr, II, III, IV, V) from the last token
- Removes diacritics (accents) and normalizes whitespace/punctuation
- Handles dotted/abbrev football names like "K.Murray"
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional


_SUFFIXES: set[str] = {"jr", "sr", "ii", "iii", "iv", "v"}


def _strip_accents(s: str) -> str:
    if not s:
        return ""
    # Normalize to NFKD and drop combining marks
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _tokens(s: str) -> list[str]:
    if not s:
        return []
    # Replace common punctuation with space, keep apostrophes as letters are merged later
    t = s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    t = re.sub(r"[\.,\-_]", " ", t)
    t = _normalize_ws(t)
    return [p for p in t.split(" ") if p]


def normalize_name(s: str) -> str:
    """Strict normalization: lower, trim, single space; keep letters/digits/punc.
    Useful for human-readable comparisons.
    """
    t = _strip_accents(str(s or "")).lower()
    return _normalize_ws(t)


def normalize_name_loose(s: str) -> str:
    """Loose normalization: lower, keep only letters/digits.
    Collapses variants like "K.Murray" and "Kyler Murray" -> "kylermurray".
    """
    t = _strip_accents(str(s or "")).lower()
    return re.sub(r"[^a-z0-9]", "", t)


def _drop_suffix(tok: str) -> str:
    t = re.sub(r"[^a-z0-9]", "", tok.lower())
    return "" if t in _SUFFIXES else tok


def normalize_alias_init_last(s: str) -> str:
    """Alias built from first initial + last name, alphanumeric and lower.

    Examples:
    - "Kyler Murray" -> "kmurray"
    - "K.Murray" -> "kmurray"
    - "Odell Beckham Jr." -> "obeckham"
    - "Amon-Ra St. Brown" -> "abrown"
    - "Marquez Valdes-Scantling" -> "mvaldesscantling"
    """
    raw = str(s or "").strip()
    if not raw:
        return ""
    raw = _strip_accents(raw)
    parts = _tokens(raw)
    if not parts:
        return ""
    # First initial from first token's first letter/digit
    first_initial = re.sub(r"[^a-z0-9]", "", parts[0].lower())[:1]
    if not first_initial and len(parts[0]) > 0:
        first_initial = parts[0][0:1].lower()
    # Determine last token, dropping suffixes if present
    last_idx = len(parts) - 1
    while last_idx >= 0 and _drop_suffix(parts[last_idx]) == "":
        last_idx -= 1
    if last_idx < 0:
        last_tok = parts[-1]
        last2_tok = ""
    else:
        last_tok = parts[last_idx]
        last2_tok = parts[last_idx - 1] if last_idx - 1 >= 1 else ""
    # Joiner particles that are part of many multi-word last names
    joiners = {"st", "st.", "de", "del", "da", "di", "van", "von", "la", "le", "mc", "mac", "o"}
    # If original raw contains a hyphen in the tail or the previous token is a joiner, combine last two tokens
    raw_tail = " ".join(parts[max(1, last_idx - 1):]).lower() if last_idx >= 0 else raw.lower()
    combine = ("-" in raw.lower()) or (re.sub(r"[^a-z0-9]", "", str(last2_tok).lower()) in joiners)
    last_clean = re.sub(r"[^a-z0-9]", "", str(last_tok).lower())
    if combine and last2_tok:
        prev_clean = re.sub(r"[^a-z0-9]", "", str(last2_tok).lower())
        last_clean = f"{prev_clean}{last_clean}"
    return f"{first_initial}{last_clean}"


__all__ = [
    "normalize_name",
    "normalize_name_loose",
    "normalize_alias_init_last",
]

# --- Canonical alias mapping for known nicknames ---
# Keys should be lowercase strict names; values are preferred display names
_CANONICAL_NAME_ALIASES: dict[str, str] = {
    # Hollywood Brown -> Marquise Brown
    "hollywood brown": "Marquise Brown",
    # Common variants (defensive, in case upstream sends dotted/football names)
    "hollywoodbrown": "Marquise Brown",
}


def canonical_player_name(name: Optional[str]) -> str:
    """Return a canonical display name for a player if a known alias is used.

    - Keeps original name when no mapping exists
    - Strips accents and normalizes whitespace before matching
    """
    if not name:
        return ""
    raw = _normalize_ws(_strip_accents(str(name)))
    key_strict = raw.lower()
    if key_strict in _CANONICAL_NAME_ALIASES:
        return _CANONICAL_NAME_ALIASES[key_strict]
    # Try loose key (alnum only)
    key_loose = normalize_name_loose(raw)
    if key_loose in _CANONICAL_NAME_ALIASES:
        return _CANONICAL_NAME_ALIASES[key_loose]
    return raw

__all__.append("canonical_player_name")
