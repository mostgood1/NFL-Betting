import json
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LINES_CSV = DATA_DIR / "lines.csv"
OUT_CSV = LINES_CSV  # in-place update


def _load_snapshots() -> Dict[str, pd.DataFrame]:
    """Load all real_betting_lines_*.json snapshots into a mapping by date string.
    Returns {date_str: df}, where df has columns ['home_team','away_team','spread_home','total'] (and prices if present).
    """
    snaps: Dict[str, pd.DataFrame] = {}
    for fp in sorted(DATA_DIR.glob("real_betting_lines_*.json")):
        try:
            date_key = fp.stem.replace("real_betting_lines_", "").replace("-", "_")
            blob = json.loads(fp.read_text(encoding="utf-8"))
            # Reuse parser logic from data_sources by importing lazily to avoid heavy deps
            from nfl_compare.src.data_sources import _parse_real_lines_json  # type: ignore
            df = _parse_real_lines_json(blob)
            if not df.empty:
                # Normalize float types
                for c in ["spread_home", "total"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                snaps[date_key] = df
        except Exception:
            continue
    return snaps


def backfill_close_fields(lines: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Populate close_spread_home and close_total when missing using snapshots or fallbacks.

    Strategy:
    1) For each row lacking close_* fields, try to find the last snapshot on or before the game date
       that matches (home_team, away_team) and use its spread_home/total as close values.
    2) If still missing, and if the CSV has final market 'spread_home'/'total' values (non-null), use them.
    3) Record simple counts and return the updated DataFrame and a report dict.
    """
    if lines.empty:
        return lines, {"rows": 0, "updated_spread": 0, "updated_total": 0}

    # Ensure expected cols
    for c in ["close_spread_home", "close_total", "game_date", "date"]:
        if c not in lines.columns:
            lines[c] = pd.NA

    # Build a simple date key for snapshot lookup
    date_col = "game_date" if "game_date" in lines.columns else ("date" if "date" in lines.columns else None)
    if date_col and lines[date_col].notna().any():
        dates = pd.to_datetime(lines[date_col], errors="coerce").dt.strftime("%Y_%m_%d")
    else:
        dates = pd.Series([None] * len(lines), index=lines.index)

    snaps = _load_snapshots()
    updated_spread = 0
    updated_total = 0

    for idx, row in lines.iterrows():
        if pd.isna(row.get("close_spread_home")) or pd.isna(row.get("close_total")):
            # choose the nearest snapshot on or before date
            date_key = dates.loc[idx]
            df_snap = None
            if isinstance(date_key, str) and date_key in snaps:
                df_snap = snaps[date_key]
            else:
                # try any earlier snapshot (sorted keys), pick the last one
                keys = [k for k in snaps.keys() if isinstance(date_key, str) and k <= date_key]
                if keys:
                    df_snap = snaps[sorted(keys)[-1]]
            if df_snap is not None and not df_snap.empty:
                mask = (
                    df_snap["home_team"].astype(str).str.lower() == str(row.get("home_team", "")).lower()
                ) & (
                    df_snap["away_team"].astype(str).str.lower() == str(row.get("away_team", "")).lower()
                )
                cand = df_snap[mask]
                if not cand.empty:
                    if pd.isna(row.get("close_spread_home")) and "spread_home" in cand.columns:
                        v = pd.to_numeric(cand.iloc[0]["spread_home"], errors="coerce")
                        if pd.notna(v):
                            lines.at[idx, "close_spread_home"] = float(v)
                            updated_spread += 1
                    if pd.isna(row.get("close_total")) and "total" in cand.columns:
                        v = pd.to_numeric(cand.iloc[0]["total"], errors="coerce")
                        if pd.notna(v):
                            lines.at[idx, "close_total"] = float(v)
                            updated_total += 1
            # Fallback to current row's market fields if still missing
            if pd.isna(lines.at[idx, "close_spread_home"]) and pd.notna(row.get("spread_home")):
                try:
                    lines.at[idx, "close_spread_home"] = float(row.get("spread_home"))
                    updated_spread += 1
                except Exception:
                    pass
            if pd.isna(lines.at[idx, "close_total"]) and pd.notna(row.get("total")):
                try:
                    lines.at[idx, "close_total"] = float(row.get("total"))
                    updated_total += 1
                except Exception:
                    pass

    report = {
        "rows": int(len(lines)),
        "updated_spread": int(updated_spread),
        "updated_total": int(updated_total),
        "out_path": str(OUT_CSV),
    }
    return lines, report


if __name__ == "__main__":
    if not LINES_CSV.exists():
        print(json.dumps({"status": "error", "error": "lines.csv not found", "path": str(LINES_CSV)}))
        raise SystemExit(1)
    df = pd.read_csv(LINES_CSV)
    df2, report = backfill_close_fields(df)
    # write in-place
    df2.to_csv(OUT_CSV, index=False)
    print(json.dumps({"status": "ok", **report}))
