"""
Seed/enrich lines.csv for a specific season/week using the latest real odds snapshot.

Usage:
  python scripts/seed_lines_for_week.py --season 2025 --week 4
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

def _ensure_repo_on_path() -> None:
    # When executed as a script from scripts/, parent (repo root) isn't on sys.path.
    # Add it so 'nfl_compare' package can be imported.
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))

def main() -> None:
    _ensure_repo_on_path()
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    args = ap.parse_args()

    # Lazy imports from package to avoid heavy module import at CLI start
    from nfl_compare.src.data_sources import DATA_DIR as _DATA_DIR
    from nfl_compare.src.data_sources import load_games
    from nfl_compare.src.team_normalizer import normalize_team_name
    try:
        # private helper is okay for internal script usage
        from nfl_compare.src.data_sources import _try_load_latest_real_lines as _latest_json  # type: ignore
    except Exception as e:
        print(f"Error: cannot import odds JSON loader: {e}")
        return

    games = load_games()
    if games is None or games.empty:
        print("No games.csv rows found; aborting.")
        return

    # Normalize for matching
    games['home_team'] = games['home_team'].astype(str).apply(normalize_team_name)
    games['away_team'] = games['away_team'].astype(str).apply(normalize_team_name)

    g_slice = games[(games.get('season').astype('Int64') == int(args.season)) & (games.get('week').astype('Int64') == int(args.week))].copy()
    if g_slice is None or g_slice.empty:
        print(f"No games found for season={args.season}, week={args.week}; nothing to seed.")
        return

    lines_fp = _DATA_DIR / 'lines.csv'
    if lines_fp.exists():
        try:
            lines_df = pd.read_csv(lines_fp)
        except Exception:
            lines_df = pd.DataFrame()
    else:
        # Create empty with expected columns
        try:
            from nfl_compare.src.schemas import _field_names, LineRow  # type: ignore
            lines_df = pd.DataFrame(columns=_field_names(LineRow))
        except Exception:
            # Minimal set
            lines_df = pd.DataFrame(columns=['season','week','game_id','home_team','away_team','spread_home','total','moneyline_home','moneyline_away','close_spread_home','close_total','date'])

    if 'home_team' in lines_df.columns:
        lines_df['home_team'] = lines_df['home_team'].astype(str).apply(normalize_team_name)
    if 'away_team' in lines_df.columns:
        lines_df['away_team'] = lines_df['away_team'].astype(str).apply(normalize_team_name)

    # Seed new rows for missing game_ids (or home/away pairs if game_id absent)
    have_ids = set(str(x) for x in (lines_df['game_id'].dropna().astype(str).unique() if 'game_id' in lines_df.columns else []))
    have_pairs = set((str(r.get('home_team','')).lower(), str(r.get('away_team','')).lower()) for _, r in (lines_df[['home_team','away_team']] if {'home_team','away_team'}.issubset(lines_df.columns) else pd.DataFrame(columns=['home_team','away_team'])).iterrows())

    add = []
    for _, r in g_slice.iterrows():
        gid = str(r.get('game_id')) if pd.notna(r.get('game_id')) else ''
        home = str(r.get('home_team',''))
        away = str(r.get('away_team',''))
        pair = (home.lower(), away.lower())
        need = False
        if gid:
            need = gid not in have_ids
        else:
            need = pair not in have_pairs
        if need:
            add.append({
                'season': int(r.get('season')) if pd.notna(r.get('season')) else None,
                'week': int(r.get('week')) if pd.notna(r.get('week')) else None,
                'game_id': gid or None,
                'home_team': home,
                'away_team': away,
                'spread_home': pd.NA,
                'total': pd.NA,
                'moneyline_home': pd.NA,
                'moneyline_away': pd.NA,
                'close_spread_home': pd.NA,
                'close_total': pd.NA,
                'date': r.get('date'),
            })
    if add:
        lines_df = pd.concat([lines_df, pd.DataFrame(add)], ignore_index=True)

    # Enrich with latest JSON odds
    try:
        j = _latest_json()
    except Exception:
        j = pd.DataFrame()

    if j is not None and not j.empty:
        from nfl_compare.src.team_normalizer import normalize_team_name
        j['home_team'] = j['home_team'].astype(str).apply(normalize_team_name)
        j['away_team'] = j['away_team'].astype(str).apply(normalize_team_name)
        key = ['home_team','away_team']
        cols = ['spread_home','total','moneyline_home','moneyline_away','spread_home_price','spread_away_price','total_over_price','total_under_price']
        merge_cols = [c for c in cols if c in j.columns]
        merged = lines_df.merge(j[key+merge_cols], on=key, how='left', suffixes=('', '_json'))
        for c in merge_cols:
            jc = f'{c}_json'
            if jc in merged.columns:
                merged[c] = merged[c].where(merged[c].notna(), merged[jc])
        drop = [c for c in merged.columns if c.endswith('_json')]
        if drop:
            merged = merged.drop(columns=drop)
        lines_df = merged

    # Write back
    lines_fp.parent.mkdir(parents=True, exist_ok=True)
    lines_df.to_csv(lines_fp, index=False)
    print(f"Seeded/enriched lines.csv for season={args.season} week={args.week}; rows={len(lines_df)} written → {lines_fp}")


if __name__ == "__main__":
    import pandas as pd  # ensure pd in scope used above
    main()
