from __future__ import annotations

"""
Consensus market aggregator for game lines.

Combines available snapshots/sources to compute a consensus spread/total/moneyline
for each game in a given weekly view. Designed to be defensive and additive: it only
fills market fields when missing and never overwrites existing close lines.

Sources considered (best-effort):
  - OddsAPI unified snapshots written as real_betting_lines_*.json via data_sources._try_load_latest_real_lines
  - Bovada game markets CSV if present for (season, week): bovada_game_props_<season>_wk<week>.csv

Output columns added to the provided view DataFrame:
  - cons_spread_home, cons_total, cons_moneyline_home, cons_moneyline_away
  - cons_spread_home_price, cons_spread_away_price, cons_total_over_price, cons_total_under_price
  - cons_source_count: number of sources contributing to that row

Additionally, this helper will fill market_spread_home and market_total when they are
missing by using consensus values, but will not overwrite close_* fields or already
populated market_*.
"""

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from .data_sources import _try_load_latest_real_lines
from .team_normalizer import normalize_team_name as _norm_team

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'


def _latest_bovada_game_csv(season: Optional[int], week: Optional[int]) -> Optional[Path]:
    try:
        if season is None or week is None:
            return None
        fp = DATA_DIR / f'bovada_game_props_{season}_wk{int(week)}.csv'
        if fp.exists():
            return fp
        # fallback: try multiple weeks (rare)
        matches = sorted(DATA_DIR.glob(f'bovada_game_props_{season}_wk*.csv'))
        return matches[-1] if matches else None
    except Exception:
        return None


def _extract_bovada_main_game_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Extract a single row per (home, away) with primary game lines from Bovada CSV.
    We expect columns: market_key, period, is_alternate, line, price_home, price_away, over_price, under_price.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    b = df.copy()
    for col in ('home_team','away_team','market_key','period'):
        if col in b.columns:
            b[col] = b[col].astype(str).apply(_norm_team)
    # Keep only full game markets
    if 'period' in b.columns:
        b = b[b['period'].astype(str).str.upper().eq('G')]
    # Remove alternates
    if 'is_alternate' in b.columns:
        b = b[(b['is_alternate'] == False) | b['is_alternate'].isna()]
    # Spread
    spread = None
    try:
        sp = b[b['market_key'].astype(str).str.lower().str.contains('spread')].copy()
        # Bovada stores team_side rows; derive home line/price where possible
        if 'team_side' in sp.columns:
            sp_home = sp[sp['team_side'].astype(str).str.lower().eq('home')]
        else:
            sp_home = sp
        sp_home = sp_home.rename(columns={'line':'spread_home', 'price_home':'spread_home_price', 'price_away':'spread_away_price'})
        spread = sp_home[['home_team','away_team','spread_home','spread_home_price','spread_away_price']]
    except Exception:
        spread = None
    # Total
    total = None
    try:
        tt = b[b['market_key'].astype(str).str.lower().str.contains('total')].copy()
        tt = tt.rename(columns={'line':'total', 'over_price':'total_over_price', 'under_price':'total_under_price'})
        # Drop variant keys like team_total
        tt = tt[tt['market_key'].astype(str).str.lower().eq('total')]
        total = tt[['home_team','away_team','total','total_over_price','total_under_price']]
    except Exception:
        total = None
    # Moneyline
    money = None
    try:
        ml = b[b['market_key'].astype(str).str.lower().eq('moneyline')].copy()
        ml = ml.rename(columns={'price_home':'moneyline_home', 'price_away':'moneyline_away'})
        money = ml[['home_team','away_team','moneyline_home','moneyline_away']]
    except Exception:
        money = None

    parts = []
    for d in (spread, total, money):
        if d is not None and not d.empty:
            parts.append(d)
    if not parts:
        return pd.DataFrame()
    out = parts[0]
    for d in parts[1:]:
        try:
            out = out.merge(d, on=['home_team','away_team'], how='outer')
        except Exception:
            pass
    return out


def build_consensus_for_view(view_df: pd.DataFrame, season: Optional[int], week: Optional[int]) -> pd.DataFrame:
    if view_df is None or view_df.empty:
        return view_df
    v = view_df.copy()
    # Normalize team names for join
    for col in ('home_team','away_team'):
        if col in v.columns:
            v[col] = v[col].astype(str).apply(_norm_team)

    # Source A: OddsAPI latest snapshot (simple lines table)
    try:
        odds = _try_load_latest_real_lines()
        if odds is not None and not odds.empty:
            for col in ('home_team','away_team'):
                if col in odds.columns:
                    odds[col] = odds[col].astype(str).apply(_norm_team)
    except Exception:
        odds = pd.DataFrame()

    # Source B: Bovada game CSV for given week
    try:
        bfp = _latest_bovada_game_csv(season, week)
        bov = pd.read_csv(bfp) if (bfp and bfp.exists()) else pd.DataFrame()
        bov_main = _extract_bovada_main_game_lines(bov) if not bov.empty else pd.DataFrame()
    except Exception:
        bov_main = pd.DataFrame()

    # Merge sources into a single per-game table keyed by (home, away)
    def _coalesce(a: Optional[pd.Series], b: Optional[pd.Series]) -> pd.Series:
        if a is None or a.empty:
            return b if b is not None else pd.Series(index=v.index, data=np.nan)
        if b is None or b.empty:
            return a
        a2 = pd.to_numeric(a, errors='coerce')
        b2 = pd.to_numeric(b, errors='coerce')
        return a2.where(a2.notna(), b2)

    # Aggregate candidates per view row by joining
    merged = v.copy()
    sources_used = 0
    if odds is not None and not odds.empty:
        sources_used += 1
        try:
            merged = merged.merge(odds, on=['home_team','away_team'], how='left', suffixes=('', '_odds'))
        except Exception:
            pass
    if bov_main is not None and not bov_main.empty:
        sources_used += 1
        try:
            merged = merged.merge(bov_main, on=['home_team','away_team'], how='left', suffixes=('', '_bov'))
        except Exception:
            pass

    # Compute consensus lines using medians across available sources (odds snapshot and bovada)
    def _median_row(row: pd.Series, names: list[str]) -> float:
        vals = []
        for n in names:
            if n in row.index and pd.notna(row[n]):
                try:
                    vals.append(float(row[n]))
                except Exception:
                    pass
        if not vals:
            return np.nan
        return float(np.median(vals))

    cols = merged.columns
    # Determine candidate column names prefixed by source suffixes
    spread_candidates = [c for c in ['spread_home','spread_home_odds','spread_home_bov'] if c in cols]
    total_candidates = [c for c in ['total','total_odds','total_bov'] if c in cols]
    mlh_candidates = [c for c in ['moneyline_home','moneyline_home_odds','moneyline_home_bov'] if c in cols]
    mla_candidates = [c for c in ['moneyline_away','moneyline_away_odds','moneyline_away_bov'] if c in cols]
    shp_candidates = [c for c in ['spread_home_price','spread_home_price_odds','spread_home_price_bov'] if c in cols]
    sap_candidates = [c for c in ['spread_away_price','spread_away_price_odds','spread_away_price_bov'] if c in cols]
    top_candidates = [c for c in ['total_over_price','total_over_price_odds','total_over_price_bov'] if c in cols]
    tup_candidates = [c for c in ['total_under_price','total_under_price_odds','total_under_price_bov'] if c in cols]

    merged['cons_spread_home'] = merged.apply(lambda r: _median_row(r, spread_candidates), axis=1) if spread_candidates else np.nan
    merged['cons_total'] = merged.apply(lambda r: _median_row(r, total_candidates), axis=1) if total_candidates else np.nan
    merged['cons_moneyline_home'] = merged.apply(lambda r: _median_row(r, mlh_candidates), axis=1) if mlh_candidates else np.nan
    merged['cons_moneyline_away'] = merged.apply(lambda r: _median_row(r, mla_candidates), axis=1) if mla_candidates else np.nan
    merged['cons_spread_home_price'] = merged.apply(lambda r: _median_row(r, shp_candidates), axis=1) if shp_candidates else np.nan
    merged['cons_spread_away_price'] = merged.apply(lambda r: _median_row(r, sap_candidates), axis=1) if sap_candidates else np.nan
    merged['cons_total_over_price'] = merged.apply(lambda r: _median_row(r, top_candidates), axis=1) if top_candidates else np.nan
    merged['cons_total_under_price'] = merged.apply(lambda r: _median_row(r, tup_candidates), axis=1) if tup_candidates else np.nan
    merged['cons_source_count'] = sources_used

    # Fill market_* fields if missing using consensus, but never overwrite close_* if present
    try:
        if 'market_spread_home' not in merged.columns:
            merged['market_spread_home'] = np.nan
        if 'market_total' not in merged.columns:
            merged['market_total'] = np.nan
        m_spread = merged['market_spread_home'].isna() & pd.notna(merged['cons_spread_home'])
        merged.loc[m_spread, 'market_spread_home'] = merged.loc[m_spread, 'cons_spread_home']
        m_total = merged['market_total'].isna() & pd.notna(merged['cons_total'])
        merged.loc[m_total, 'market_total'] = merged.loc[m_total, 'cons_total']
    except Exception:
        pass

    return merged
