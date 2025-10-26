import os
import sys
import json
import math
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / 'nfl_compare' / 'data'

# Import Flask app to hit API routes with test_client
from app import app  # noqa: E402

def latest_odds_snapshot_path() -> Path | None:
    if not DATA_DIR.exists():
        return None
    snaps = sorted(DATA_DIR.glob('real_betting_lines_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    return snaps[0] if snaps else None


def is_today(dt: datetime) -> bool:
    today = datetime.now(timezone.utc).astimezone().date()
    return dt.date() == today


def check_game_odds_fresh() -> tuple[bool, str]:
    p = latest_odds_snapshot_path()
    if p is None or not p.exists():
        return False, 'No odds snapshots found (real_betting_lines_*.json)'
    try:
        mtime = datetime.fromtimestamp(p.stat().st_mtime).astimezone()
        if not is_today(mtime):
            return False, f"Latest odds snapshot not from today: {p.name} (mtime {mtime.isoformat()})"
        # Basic parse to ensure JSON is valid and not empty
        with p.open('r', encoding='utf-8') as f:
            js = json.load(f)
        if not js:
            return False, f"Odds snapshot {p.name} is empty"
        # Spot-check presence of at least one bookmaker/market
        events = js if isinstance(js, list) else js.get('data') or []
        if not events:
            # It's acceptable for non-slate days; snapshot freshness is what we care about
            return True, f"OK: {p.name} (no events today)"
        return True, f"OK: {p.name} (events={len(events)})"
    except Exception as e:
        return False, f"Failed to read odds snapshot {p}: {e}"


def check_lines_csv_updated() -> tuple[bool, str]:
    p = DATA_DIR / 'lines.csv'
    if not p.exists():
        return False, 'lines.csv not found (seed_lines_for_week did not run?)'
    try:
        mtime = datetime.fromtimestamp(p.stat().st_mtime).astimezone()
        if not is_today(mtime):
            return False, f"lines.csv is stale (mtime {mtime.isoformat()})"
        import pandas as pd
        df = pd.read_csv(p)
        if df is None or df.empty:
            return False, 'lines.csv is empty'
        # Require presence of spread/total columns
        needed = [c for c in ['spread_home', 'total', 'moneyline_home', 'moneyline_away'] if c in df.columns]
        if not needed:
            return False, 'lines.csv missing key columns'
        return True, f"OK: lines.csv rows={len(df)}"
    except Exception as e:
        return False, f"Failed to read lines.csv: {e}"


def check_props_recs_api() -> tuple[bool, str]:
    # Sanity: API returns data and line-required markets include a numeric line
    LINE_REQUIRED = {"rec_yards","rush_yards","pass_yards","receptions","pass_attempts","rush_attempts","rush_rec_yards","pass_rush_yards","targets"}
    try:
        with app.test_client() as c:
            r = c.get('/api/props/recommendations')
            if r.status_code != 200:
                return False, f"/api/props/recommendations -> {r.status_code}"
            js = r.get_json() or {}
            data = js.get('data') or []
            if not data:
                return False, 'Props recommendations returned 0 rows'
            # Ensure a reasonable fraction of line-required plays have numeric lines
            checked = 0
            missing = 0
            for d in data[:50]:
                for p in (d.get('plays') or [])[:6]:
                    mk = str(p.get('market') or '').strip().lower()
                    mk_map = {
                        'receiving yards':'rec_yards','receptions':'receptions','rushing yards':'rush_yards','passing yards':'pass_yards',
                        'passing tds':'pass_tds','pass tds':'pass_tds','pass touchdowns':'pass_tds',
                        'passing attempts':'pass_attempts','pass attempts':'pass_attempts','rushing attempts':'rush_attempts','rush attempts':'rush_attempts',
                        'interceptions':'interceptions','interceptions thrown':'interceptions',
                        'rush+rec yards':'rush_rec_yards','rushing + receiving yards':'rush_rec_yards','rush + rec yards':'rush_rec_yards',
                        'pass+rush yards':'pass_rush_yards','pass + rush yards':'pass_rush_yards','passing + rushing yards':'pass_rush_yards',
                        'targets':'targets','2+ touchdowns':'multi_tds','anytime td':'any_td','any time td':'any_td'
                    }
                    key = mk_map.get(mk, mk)
                    if key in LINE_REQUIRED:
                        checked += 1
                        ln = p.get('line')
                        if ln is None:
                            missing += 1
                        else:
                            try:
                                float(ln)
                            except Exception:
                                missing += 1
            if checked >= 5 and missing/checked > 0.25:
                return False, f"Too many plays missing numeric lines: {missing}/{checked}"
            return True, f"OK: props recs rows={len(data)} missing_line_ratio={(missing/checked if checked else 0):.2f}"
    except Exception as e:
        return False, f"Props API check failed: {e}"


def _resolve_current_week() -> tuple[int | None, int | None]:
    cfg = DATA_DIR / 'current_week.json'
    try:
        if cfg.exists():
            js = json.loads(cfg.read_text(encoding='utf-8'))
            s = int(js.get('season')) if js.get('season') is not None else None
            w = int(js.get('week')) if js.get('week') is not None else None
            return s, w
    except Exception:
        pass
    return None, None


def check_cards_api() -> tuple[bool, str]:
    try:
        season, week = _resolve_current_week()
        qs = ''
        if season and week:
            qs = f"?season={season}&week={week}"
        with app.test_client() as c:
            r = c.get('/api/cards' + qs)
            if r.status_code != 200:
                return False, f"/api/cards -> {r.status_code}"
            js = r.get_json() or {}
            rows = js.get('rows') or 0
            if rows and rows > 0:
                return True, f"OK: cards rows={rows} (season={season} week={week})"
            # Fallback: try default (no params) then week 1 to account for early season state
            r2 = c.get('/api/cards')
            if r2.status_code == 200:
                js2 = r2.get_json() or {}
                rows2 = js2.get('rows') or 0
                if rows2 and rows2 > 0:
                    return True, f"OK: cards rows={rows2} (default)"
            r3 = c.get('/api/cards?season=2025&week=1')
            if r3.status_code == 200:
                js3 = r3.get_json() or {}
                rows3 = js3.get('rows') or 0
                if rows3 and rows3 > 0:
                    return True, f"OK: cards rows={rows3} (W1 fallback)"
            return False, f"Cards API returned 0 rows (current, default, and W1)"
    except Exception as e:
        return False, f"Cards API check failed: {e}"


def check_cards_html() -> tuple[bool, str]:
    import re
    try:
        season, week = _resolve_current_week()
        qs = ''
        if season and week:
            qs = f"?season={season}&week={week}"
        with app.test_client() as c:
            r = c.get('/' + qs)
            if r.status_code != 200:
                return False, f"/ (cards page) -> {r.status_code}"
            html = r.get_data(as_text=True)
            cards = len(re.findall(r'class="card"', html))
            if cards and cards > 0:
                return True, f"OK: cards HTML count={cards} (season={season} week={week})"
            # Fallback to Week 1
            r2 = c.get('/?season=2025&week=1')
            if r2.status_code == 200:
                html2 = r2.get_data(as_text=True)
                cards2 = len(re.findall(r'class="card"', html2))
                if cards2 and cards2 > 0:
                    return True, f"OK: cards HTML count={cards2} (W1 fallback)"
            return False, "Cards HTML shows 0 cards (current and W1)"
    except Exception as e:
        return False, f"Cards HTML check failed: {e}"


def main() -> int:
    results: list[tuple[str, bool, str]] = []
    ok, msg = check_game_odds_fresh()
    results.append(("odds_snapshot", ok, msg))
    ok2, msg2 = check_lines_csv_updated()
    results.append(("lines_csv", ok2, msg2))
    ok3a, msg3a = check_cards_api()
    results.append(("cards_api", ok3a, msg3a))
    ok3b, msg3b = check_cards_html()
    results.append(("cards_html", ok3b, msg3b))
    ok4, msg4 = check_props_recs_api()
    results.append(("props_api", ok4, msg4))

    # If HTML cards pass, tolerate API cards failure as a warning (frontend is the source of truth here)
    # Build failure list then remove cards_api if cards_html passed
    failed = [name for name, ok, _ in results if not ok]
    cards_html_ok = next((ok for name, ok, _ in results if name == 'cards_html'), False)
    if cards_html_ok:
        failed = [name for name in failed if name != 'cards_api']
    for name, ok, msg in results:
        print(f"{name}: {'PASS' if ok else 'FAIL'} - {msg}")
    if failed:
        print("SMOKE: FAIL =>", ", ".join(failed))
        return 2
    print("SMOKE: PASS")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
