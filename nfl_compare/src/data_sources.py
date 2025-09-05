import json
import warnings
from datetime import datetime
import os
from typing import Dict, Any, List
import pandas as pd
from pathlib import Path
from .schemas import GameRow, TeamStatRow, LineRow
from .team_normalizer import normalize_team_name

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Silence pandas concat FutureWarning about empty/all-NA entries impacting dtype inference
warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
    category=FutureWarning,
)

def _field_names(model_cls) -> list:
    """Return model field names for Pydantic v1 or v2."""
    try:
        return list(model_cls.model_fields.keys())  # Pydantic v2
    except AttributeError:
        try:
            return list(model_cls.__fields__.keys())  # Pydantic v1
        except Exception:
            return []

# CSV readers; swap out to real APIs when ready

def load_games() -> pd.DataFrame:
    fp = DATA_DIR / "games.csv"
    if not fp.exists():
        return pd.DataFrame(columns=_field_names(GameRow))
    df = pd.read_csv(fp)
    return df

def load_team_stats() -> pd.DataFrame:
    fp = DATA_DIR / "team_stats.csv"
    if not fp.exists():
        return pd.DataFrame(columns=_field_names(TeamStatRow))
    df = pd.read_csv(fp)
    return df

def _parse_real_lines_json(blob: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse a real betting lines JSON blob into a normalized DataFrame with columns:
    ['away_team','home_team','moneyline_home','moneyline_away','spread_home','total',
     'spread_home_price','spread_away_price','total_over_price','total_under_price']

    Supports multiple structures:
    - {'lines': { 'Away @ Home': { 'moneyline': {'home': -120, 'away': +110},
                                   'total_runs' or 'totals': {...},
                                   'run_line' or 'spreads': {...},
                                   'markets': [...] } } }
    """
    lines = blob.get('lines', {}) if isinstance(blob, dict) else {}
    rows: List[Dict[str, Any]] = []

    for matchup_key, game_lines in lines.items():
        try:
            # Expect keys like "Away Team @ Home Team"
            if '@' in matchup_key:
                away_team, home_team = [p.strip() for p in matchup_key.split('@', 1)]
            elif 'vs' in matchup_key:
                away_team, home_team = [p.strip() for p in matchup_key.split('vs', 1)]
            else:
                # Unknown key format; skip
                continue

            ml_home = None
            ml_away = None
            spread_home = None
            total_line = None
            spread_home_price = None
            spread_away_price = None
            total_over_price = None
            total_under_price = None

            # Direct moneyline structure
            if isinstance(game_lines, dict) and 'moneyline' in game_lines:
                try:
                    ml = game_lines['moneyline']
                    ml_home = ml.get('home')
                    ml_away = ml.get('away')
                except Exception:
                    pass

            # Totals (object). Some upstream producers use basebally 'total_runs', others 'totals' or 'total'
            if isinstance(game_lines, dict) and 'total_runs' in game_lines:
                tr = game_lines['total_runs'] or {}
                total_line = tr.get('line', total_line)
                # capture prices if provided
                total_over_price = tr.get('over', total_over_price)
                total_under_price = tr.get('under', total_under_price)
            # Some feeds use 'total' dict
            if isinstance(game_lines, dict) and 'total' in game_lines and isinstance(game_lines['total'], dict):
                total_line = game_lines['total'].get('line', total_line)

            # Spread (object)
            if isinstance(game_lines, dict) and 'run_line' in game_lines:
                rl = game_lines['run_line']
                # NFL: want home spread (negative if favored)
                spread_home = rl.get('home', rl.get('line', spread_home))

            # Markets array (Odds API-like)
            markets = []
            if isinstance(game_lines, dict) and 'markets' in game_lines and isinstance(game_lines['markets'], list):
                markets = game_lines['markets']
            def _is_full_game(m: Dict[str, Any]) -> bool:
                key = str(m.get('key', '')).lower()
                desc = str(m.get('description', '')).lower()
                # Exclude half/quarter markets
                bad_tokens = ['1h', '2h', 'half', 'q1', 'q2', 'q3', 'q4', 'quarter']
                return not any(t in key or t in desc for t in bad_tokens)

            for m in markets:
                if not _is_full_game(m):
                    continue
                key = m.get('key')
                outcomes = m.get('outcomes', []) or []
                if key in ('h2h', 'moneyline'):
                    for o in outcomes:
                        name = str(o.get('name', ''))
                        price = o.get('price')
                        if not name or price is None:
                            continue
                        if name.strip().lower() == home_team.lower():
                            ml_home = price
                        elif name.strip().lower() == away_team.lower():
                            ml_away = price
                elif key in ('spreads', 'spread'):
                    # pick the outcome for the home team
                    for o in outcomes:
                        name = str(o.get('name', ''))
                        pt = o.get('point')
                        price = o.get('price')
                        if name.strip().lower() == home_team.lower():
                            if pt is not None:
                                spread_home = pt
                            spread_home_price = price if price is not None else spread_home_price
                        elif name.strip().lower() == away_team.lower():
                            spread_away_price = price if price is not None else spread_away_price
                elif key in ('totals', 'total'):
                    # outcomes may include Over/Under each with same point
                    pts = [o.get('point') for o in outcomes if o.get('point') is not None]
                    if pts:
                        # choose the first consistent point
                        total_line = pts[0]
                    for o in outcomes:
                        nm = str(o.get('name','')).strip().lower()
                        price = o.get('price')
                        if nm.startswith('over'):
                            total_over_price = price if price is not None else total_over_price
                        elif nm.startswith('under'):
                            total_under_price = price if price is not None else total_under_price

            # Coerce numerics when possible
            try:
                if total_line is not None:
                    total_line = float(total_line)
            except Exception:
                pass
            try:
                if spread_home is not None:
                    spread_home = float(spread_home)
            except Exception:
                pass

            rows.append({
                'away_team': away_team,
                'home_team': home_team,
                'moneyline_home': ml_home,
                'moneyline_away': ml_away,
                'spread_home': spread_home,
                'total': total_line,
                'spread_home_price': spread_home_price,
                'spread_away_price': spread_away_price,
                'total_over_price': total_over_price,
                'total_under_price': total_under_price,
            })
        except Exception:
            # skip malformed entry
            continue

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        'away_team','home_team','moneyline_home','moneyline_away','spread_home','total',
        'spread_home_price','spread_away_price','total_over_price','total_under_price'
    ])


def _try_load_latest_real_lines() -> pd.DataFrame:
    """
    Try to load the latest real lines JSON file from ./data.
    Looks for files like real_betting_lines_YYYY_MM_DD.json, falling back to real_betting_lines.json
    Returns a normalized DataFrame or empty DataFrame.
    """
    # Allow disabling JSON odds via env for parity with Render or local debug
    if os.getenv('DISABLE_JSON_ODDS', '').strip() in ('1','true','True','yes','Y'):
        return pd.DataFrame(columns=['away_team','home_team','moneyline_home','moneyline_away','spread_home','total'])
    # Prefer today's date file
    candidates = []
    today_us = datetime.now().strftime('%Y_%m_%d')
    candidates.append(DATA_DIR / f'real_betting_lines_{today_us}.json')
    # Also try hyphenated
    today_hy = datetime.now().strftime('%Y-%m-%d')
    candidates.append(DATA_DIR / f'real_betting_lines_{today_hy}.json')
    # Generic
    candidates.append(DATA_DIR / 'real_betting_lines.json')

    for fp in candidates:
        if fp.exists():
            try:
                blob = json.loads(fp.read_text(encoding='utf-8'))
                return _parse_real_lines_json(blob)
            except Exception:
                continue
    # As a broader fallback, scan for any real_betting_lines_*.json and pick the latest by name
    try:
        files = sorted(DATA_DIR.glob('real_betting_lines_*.json'))
        for fp in reversed(files):
            try:
                blob = json.loads(fp.read_text(encoding='utf-8'))
                df = _parse_real_lines_json(blob)
                if not df.empty:
                    return df
            except Exception:
                continue
    except Exception:
        pass
    return pd.DataFrame(columns=['away_team','home_team','moneyline_home','moneyline_away','spread_home','total'])


def load_lines() -> pd.DataFrame:
    """
    Load lines from CSV and enhance with any available real odds JSON.
    Returns a DataFrame with at least: ['game_id','home_team','away_team','spread_home','total','moneyline_home','moneyline_away', 'close_spread_home','close_total'] where available.
    """
    # Base CSV (historical/backfill)
    csv_cols = _field_names(LineRow)
    df_csv = pd.DataFrame(columns=csv_cols)
    fp = DATA_DIR / "lines.csv"
    if fp.exists():
        try:
            df_csv = pd.read_csv(fp)
            # Normalize team names to canonical forms to improve join rates
            if 'home_team' in df_csv.columns:
                df_csv['home_team'] = df_csv['home_team'].astype(str).apply(normalize_team_name)
            if 'away_team' in df_csv.columns:
                df_csv['away_team'] = df_csv['away_team'].astype(str).apply(normalize_team_name)
        except Exception:
            df_csv = pd.DataFrame(columns=csv_cols)

    # Real JSON (for current/future)
    df_json = _try_load_latest_real_lines()

    if df_json.empty:
        return df_csv

    # Try to merge JSON odds into CSV by matching on team names
    # Prefer JSON values when CSV lacks values
    if not df_csv.empty:
        # Merge on team names; keep csv row identity including game_id/season/week
        merged = df_csv.merge(
            df_json,
            on=['home_team','away_team'],
            how='left',
            suffixes=('', '_json')
        )
        # Fill preferred columns from JSON when missing in CSV
        for col in ['spread_home','total','moneyline_home','moneyline_away','spread_home_price','spread_away_price','total_over_price','total_under_price']:
            json_col = f'{col}_json'
            if json_col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[json_col])
        # Drop helper cols
        drop_cols = [c for c in merged.columns if c.endswith('_json')]
        merged = merged.drop(columns=drop_cols)

        # Also include JSON-only games not present in CSV (by team pairing)
        try:
            key_cols = ['home_team', 'away_team']
            present = set(tuple(x) for x in merged[key_cols].astype(str).values.tolist())
            add_rows = df_json[~df_json[key_cols].astype(str).apply(tuple, axis=1).isin(present)].copy()
            if not add_rows.empty:
                for col in ['season','week','game_id','close_spread_home','close_total']:
                    if col not in add_rows.columns:
                        add_rows[col] = pd.NA
                # Align columns
                add_rows = add_rows.reindex(columns=merged.columns, fill_value=pd.NA)
                def _valid_df(d: pd.DataFrame) -> bool:
                    if d is None or d.empty:
                        return False
                    try:
                        return not d.isna().all().all()
                    except Exception:
                        return True
                to_concat = [df for df in (merged, add_rows) if _valid_df(df)]
                if len(to_concat) == 2:
                    # Align dtypes to avoid FutureWarning about dtype inference with all-NA cols
                    try:
                        for c in merged.columns:
                            if c in add_rows.columns:
                                try:
                                    add_rows[c] = add_rows[c].astype(merged[c].dtype)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    merged = pd.concat([merged, add_rows], ignore_index=True)
                elif len(to_concat) == 1:
                    merged = to_concat[0]
        except Exception:
            pass

        return merged
    else:
        # No CSV; return JSON with expected columns, add placeholders
        for col in ['season','week','game_id','close_spread_home','close_total']:
            df_json[col] = pd.NA
        # Reorder columns to csv_cols where possible
        return df_json[[c for c in csv_cols if c in df_json.columns] + [c for c in df_json.columns if c not in csv_cols]]

