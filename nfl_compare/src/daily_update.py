from __future__ import annotations

"""
Daily update runner:
- Loads .env
- Fetches real NFL odds (moneyline/spreads/totals) and writes today's JSON
- Updates weather snapshots for all future game dates
- Re-runs predictions to incorporate fresh odds + weather

Usage (from nfl_compare):
  python -m src.daily_update
"""

from pathlib import Path
from datetime import date as _date, datetime

from .config import load_env
from .odds_api_client import main as fetch_odds
from .auto_update import main as update_weather_and_predict
from .data_sources import DATA_DIR as DATA_DIR, _try_load_latest_real_lines, load_games
from .team_normalizer import normalize_team_name
import pandas as pd


def _ensure_lines_schema(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'season','week','game_id','date','home_team','away_team',
        'spread_home','total','moneyline_home','moneyline_away',
        'spread_home_price','spread_away_price','total_over_price','total_under_price',
        'close_spread_home','close_total'
    ]
    out = df.copy() if df is not None else pd.DataFrame()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[cols]


def _enrich_lines_from_latest_json() -> None:
    """Lightweight enrichment: merge latest JSON odds into lines.csv by (home, away).
    - Seeds missing rows for upcoming games from games.csv when possible.
    - Overwrites market columns with current JSON snapshot values (spread/total/moneylines/prices).
    Safe to run frequently; idempotent per snapshot.
    """
    try:
        j = _try_load_latest_real_lines()
    except Exception as e:
        print(f"JSON odds load failed: {e}")
        return
    if j is None or j.empty:
        print('No JSON odds snapshot found; skipping lines enrichment.')
        return
    # Normalize team names in JSON frame
    try:
        j['home_team'] = j['home_team'].astype(str).apply(normalize_team_name)
        j['away_team'] = j['away_team'].astype(str).apply(normalize_team_name)
    except Exception:
        pass

    # Load existing lines.csv (or create empty with schema)
    lines_fp = DATA_DIR / 'lines.csv'
    if lines_fp.exists():
        try:
            lines = pd.read_csv(lines_fp)
        except Exception:
            lines = pd.DataFrame()
    else:
        lines = pd.DataFrame()
    lines = _ensure_lines_schema(lines)
    try:
        lines['home_team'] = lines['home_team'].astype(str).apply(normalize_team_name)
        lines['away_team'] = lines['away_team'].astype(str).apply(normalize_team_name)
    except Exception:
        pass

    # Seed missing rows from games.csv for upcoming games to ensure coverage
    try:
        g = load_games()
        if g is not None and not g.empty:
            g2 = g.copy()
            g2['home_team'] = g2['home_team'].astype(str).apply(normalize_team_name)
            g2['away_team'] = g2['away_team'].astype(str).apply(normalize_team_name)
            # Consider games with missing finals (future/unplayed) to avoid seeding historical duplicates
            try:
                g2['has_final'] = (~g2['home_score'].isna()) & (~g2['away_score'].isna())
            except Exception:
                g2['has_final'] = False
            seed = g2[g2['has_final'] == False][['season','week','game_id','date','home_team','away_team']].dropna(subset=['home_team','away_team'])
            # Add rows for (home, away) not present in lines
            have_pairs = set(zip(lines['home_team'].astype(str), lines['away_team'].astype(str)))
            add_rows = []
            for _, r in seed.iterrows():
                pair = (str(r['home_team']), str(r['away_team']))
                if pair not in have_pairs:
                    add_rows.append({
                        'season': r.get('season'), 'week': r.get('week'), 'game_id': r.get('game_id'), 'date': r.get('date'),
                        'home_team': r['home_team'], 'away_team': r['away_team'],
                        'spread_home': pd.NA, 'total': pd.NA, 'moneyline_home': pd.NA, 'moneyline_away': pd.NA,
                        'spread_home_price': pd.NA, 'spread_away_price': pd.NA, 'total_over_price': pd.NA, 'total_under_price': pd.NA,
                        'close_spread_home': pd.NA, 'close_total': pd.NA,
                    })
            if add_rows:
                lines = pd.concat([lines, pd.DataFrame(add_rows)], ignore_index=True)
    except Exception as e:
        print(f"Seeding from games.csv skipped: {e}")

    # Merge current JSON odds onto lines by (home_team, away_team)
    # IMPORTANT: only update future/unplayed games to avoid overwriting saved odds for completed games.
    key = ['home_team','away_team']
    cols = ['spread_home','total','moneyline_home','moneyline_away','spread_home_price','spread_away_price','total_over_price','total_under_price']
    try:
        # Build a mask for future/unplayed games
        today = pd.Timestamp.today().normalize()
        dts = pd.to_datetime(lines.get('date'), errors='coerce')
        has_home_final = lines.get('home_score') if 'home_score' in lines.columns else pd.Series([pd.NA]*len(lines))
        has_away_final = lines.get('away_score') if 'away_score' in lines.columns else pd.Series([pd.NA]*len(lines))
        # Unplayed if scores are NA; also treat games dated today or later as eligible
        unplayed = has_home_final.isna() | has_away_final.isna()
        future_or_today = dts.isna() | (dts.dt.normalize() >= today)
        mask = (unplayed | future_or_today)

        base_cols = list(lines.columns)
        # Split frames to update only masked rows
        lines['_row_idx'] = range(len(lines))
        keep_df = lines[~mask].copy()
        upd_df = lines[mask].copy()
        # Drop market columns in update slice before merge to ensure overwrite
        upd_df = upd_df.drop(columns=[c for c in cols if c in upd_df.columns])
        merged = upd_df.merge(j[key+cols], on=key, how='left')
        # Reassemble
        combined = pd.concat([keep_df, merged], ignore_index=True)
        # Restore original column order (plus any new cols at end) and original-like ordering by _row_idx
        if '_row_idx' in combined.columns:
            combined = combined.sort_values('_row_idx').drop(columns=['_row_idx'])
        # Ensure we keep at least the original base_cols ordering
        for c in base_cols:
            if c not in combined.columns:
                combined[c] = pd.NA
        lines = combined
    except Exception as e:
        print(f"JSON odds merge failed: {e}")

    # Save back
    try:
        lines = _ensure_lines_schema(lines)
        lines.to_csv(lines_fp, index=False)
        print(f"Enriched lines.csv with latest JSON odds — rows={len(lines)}")
    except Exception as e:
        print(f"Failed writing lines.csv: {e}")


def main() -> None:
    load_env()
    # 1) Fetch odds first so predictions include latest markets
    try:
        fetch_odds()
    except Exception as e:
        print(f"Odds update failed: {e}")

    # 2) Update weather and re-run predictions (auto_update calls predict at end)
    try:
        update_weather_and_predict()
    except Exception as e:
        print(f"Weather/predict failed: {e}")

    # 2b) Enrich lines.csv with latest JSON odds for rapid live odds reflection
    try:
        _enrich_lines_from_latest_json()
    except Exception as e:
        print(f"Lines enrichment step failed: {e}")

    out_fp = Path(__file__).resolve().parents[1] / 'data' / 'predictions.csv'
    if out_fp.exists():
        print(f"Daily update complete. Predictions at {out_fp}")
    else:
        print("Daily update complete, but predictions file was not found.")

    # Light reconciliation: attempt to write prior-week props vs actuals CSV for server endpoint
    try:
        g = load_games()
        if g is not None and not g.empty:
            gg = g.copy()
            gg['date'] = pd.to_datetime(gg.get('date'), errors='coerce').dt.date
            seasons = sorted(gg.get('season').dropna().astype(int).unique())
            if seasons:
                season = int(seasons[-1])
                gs = gg[gg['season'] == season]
                today = _date.today()
                try:
                    cur_week = int(gs.loc[gs['date'] <= today, 'week'].dropna().astype(int).max())
                except Exception:
                    cur_week = int(gs.get('week').dropna().astype(int).max()) if not gs.empty else 1
                prior_week = max(1, cur_week - 1)
                try:
                    from .reconciliation import reconcile_props, summarize_errors  # lazy import
                except Exception as e:
                    print(f"Reconciliation unavailable (light updater): {e}")
                    raise
                try:
                    df_recon = reconcile_props(int(season), int(prior_week))
                except Exception as e:
                    print(f"Reconciliation compute failed (light updater): {e}")
                    df_recon = None
                if df_recon is not None and not df_recon.empty:
                    rfp = DATA_DIR / f"player_props_vs_actuals_{int(season)}_wk{int(prior_week)}.csv"
                    try:
                        df_recon.to_csv(rfp, index=False)
                        print(f"Reconciliation (light): wrote {rfp.name} with {len(df_recon)} rows (season={season}, week={prior_week}).")
                    except Exception as e:
                        print(f"Reconciliation write failed (light): {e}")
                    try:
                        summ = summarize_errors(df_recon)
                        if summ is not None and not summ.empty:
                            print("Reconciliation summary (light):")
                            print(summ.to_string(index=False))
                            # Persist summary to rolling history for week-over-week tracking
                            try:
                                hist_fp = DATA_DIR / 'recon_summary_history.csv'
                                summ2 = summ.copy()
                                summ2['season'] = int(season)
                                summ2['week'] = int(prior_week)
                                summ2['ts'] = datetime.utcnow().isoformat()
                                if hist_fp.exists():
                                    try:
                                        prev = pd.read_csv(hist_fp)
                                    except Exception:
                                        prev = pd.DataFrame()
                                    combined = pd.concat([prev, summ2], ignore_index=True)
                                    if set(['season','week','position']).issubset(combined.columns):
                                        combined = (
                                            combined
                                            .sort_values('ts')
                                            .drop_duplicates(subset=['season','week','position'], keep='last')
                                        )
                                    combined.to_csv(hist_fp, index=False)
                                else:
                                    summ2.to_csv(hist_fp, index=False)
                                print(f"Reconciliation (light): updated {hist_fp.name}.")
                            except Exception as e:
                                print(f"Reconciliation history write failed (light): {e}")
                    except Exception:
                        pass
                else:
                    print(f"Reconciliation (light): no rows for season={season}, week={prior_week}.")
    except Exception as e:
        print(f"Light reconciliation step skipped: {e}")


if __name__ == "__main__":
    main()
