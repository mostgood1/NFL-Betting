from __future__ import annotations

import os, sys, json, math, time, re, subprocess, hashlib, traceback, shlex, threading
from pathlib import Path
from typing import Optional, Dict, Any, List

from datetime import datetime
import pandas as pd
from flask import Flask, jsonify, render_template, request, g

BASE_DIR = Path(__file__).resolve().parent
_ENV_DATA_DIR = os.environ.get("NFL_DATA_DIR")
DATA_DIR = Path(_ENV_DATA_DIR) if _ENV_DATA_DIR else (BASE_DIR / "nfl_compare" / "data")
PRED_FILE = DATA_DIR / "predictions.csv"
PRED_WEEK_FILE = DATA_DIR / "predictions_week.csv"
LOCKED_PRED_FILE = DATA_DIR / "predictions_locked.csv"
STADIUM_META_FILE = DATA_DIR / "stadium_meta.csv"
LOCATION_OVERRIDES_FILE = DATA_DIR / "game_location_overrides.csv"
# Team assets file (logos/colors)
ASSETS_FILE = DATA_DIR / "nfl_team_assets.json"

app = Flask(__name__)

# --- Lightweight one-time logger (in-memory) ---
_log_once_keys: set[str] = set()
def _log_once(key: str, msg: str) -> None:
    try:
        if key in _log_once_keys:
            return
        _log_once_keys.add(key)
        print(msg)
    except Exception:
        pass


# --- Ultra-light health endpoints (always fast, minimal IO) ---
@app.route('/api/ping')
def api_ping():
    return jsonify({'ok': True, 'ts': int(time.time())})


@app.route('/health')
def health_root():
    return jsonify({'status': 'ok'})

# --- Background job state (admin refresh, etc.) ---
_job_state: Dict[str, Any] = {
    'running': False,
    'started_at': None,
    'ended_at': None,
    'ok': None,
    'logs': [],  # list[str] (lightweight annotations from the web process)
    'log_file': None,  # full path to updater log file (str)
}

# --- Simple in-memory cache for reconciliation results (per season/week) ---
_recon_cache: dict[tuple[int,int], dict[str, Any]] = {}


# --- Display aliases for player names (UI/API presentation only) ---
# Underlying computation can use canonical/legal names (e.g., Thomas Hockenson),
# but we prefer commonly known display names in responses.
DISPLAY_ALIASES: dict[str, str] = {
    # Vikings TE
    "Thomas Hockenson": "T.J. Hockenson",
}


def _apply_display_aliases(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if df is None or df.empty or 'player' not in df.columns:
            return df
        # Map exact player names to preferred display names
        df = df.copy()
        df['player'] = df['player'].map(lambda x: DISPLAY_ALIASES.get(x, x) if pd.notna(x) else x)
        return df
    except Exception:
        return df


# --- Request timing logs ---
@app.before_request
def _before_request_timer():
    try:
        g._t0 = time.time()
    except Exception:
        pass


@app.after_request
def _after_request_log(resp):
    try:
        t0 = getattr(g, '_t0', None)
        if t0 is not None:
            dt = int((time.time() - t0) * 1000)
            print(f"[req] {request.method} {request.path} -> {resp.status_code} in {dt}ms")
    except Exception:
        pass
    return resp


# --- Roster validation endpoints (top-level) ---
@app.route('/api/roster-validation')
def api_roster_validation():
    try:
        season = int(request.args.get('season') or 2025)
        week = int(request.args.get('week') or 2)
        summ_fp = DATA_DIR / f"roster_validation_summary_{season}_wk{week}.csv"
        if not summ_fp.exists():
            return jsonify({"error": "summary not found", "path": str(summ_fp)}), 404
        df = pd.read_csv(summ_fp)
        return jsonify({
            "season": season,
            "week": week,
            "rows": df.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/roster-validation/details')
def api_roster_validation_details():
    try:
        season = int(request.args.get('season') or 2025)
        week = int(request.args.get('week') or 2)
        det_fp = DATA_DIR / f"roster_validation_details_{season}_wk{week}.csv"
        if not det_fp.exists():
            return jsonify({"error": "details not found", "path": str(det_fp)}), 404
        df = pd.read_csv(det_fp)
        return jsonify({
            "season": season,
            "week": week,
            "rows": df.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/roster-validation')
def view_roster_validation():
    try:
        season = int(request.args.get('season') or 2025)
        week = int(request.args.get('week') or 2)
        summ_fp = DATA_DIR / f"roster_validation_summary_{season}_wk{week}.csv"
        if summ_fp.exists():
            df = pd.read_csv(summ_fp)
        else:
            df = pd.DataFrame()
        # Simple inline HTML table for quick viewing without new templates
        if df.empty:
            html = f"<h3>Roster Validation Summary</h3><p>No data for season {season}, week {week}.</p>"
        else:
            tbl = df.to_html(index=False, classes='table table-striped table-sm')
            html = f"<h3>Roster Validation Summary — Season {season}, Week {week}</h3>" + tbl
        return html
    except Exception as e:
        return f"Error: {e}", 500


def _recon_cache_get(key: tuple[int, int], force_refresh: bool = False) -> tuple[Optional[pd.DataFrame], bool]:
    """Return (df, hit) if cache entry is fresh; otherwise (None, False).
    Freshness is controlled by RECON_CACHE_TTL_SEC (default 1800 seconds).
    """
    try:
        if force_refresh:
            return None, False
        entry = _recon_cache.get(key)
        if not entry:
            # Try disk cache if enabled
            df_disk = _recon_cache_load_from_disk(key)
            if df_disk is not None:
                # Promote to memory
                _recon_cache_put(key, df_disk)
                return df_disk.copy(), True
            return None, False
        ttl = int(os.environ.get('RECON_CACHE_TTL_SEC', '1800'))
        ts = float(entry.get('ts') or 0.0)
        if ts <= 0:
            return None, False
        if (time.time() - ts) > max(0, ttl):
            # Expired; drop and miss
            try:
                _recon_cache.pop(key, None)
            except Exception:
                pass
            # Try disk cache before miss
            df_disk = _recon_cache_load_from_disk(key)
            if df_disk is not None:
                _recon_cache_put(key, df_disk)
                return df_disk.copy(), True
            return None, False
        df = entry.get('df')
        if isinstance(df, pd.DataFrame):
            return df.copy(), True
        return None, False
    except Exception:
        return None, False


def _recon_cache_put(key: tuple[int,int], df: pd.DataFrame) -> None:
    try:
        _recon_cache[key] = {"df": df.copy(), "ts": time.time()}
        _recon_cache_save_to_disk(key, df)
    except Exception:
        pass


def _recon_cache_dir() -> Path:
    try:
        base = os.environ.get('RECON_CACHE_DIR')
        if base:
            p = Path(base)
        else:
            p = DATA_DIR / 'recon_cache'
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        return DATA_DIR


def _recon_cache_fp(key: tuple[int,int]) -> Path:
    s, w = key
    return _recon_cache_dir() / f"recon_{int(s)}_wk{int(w)}.csv"


def _recon_cache_load_from_disk(key: tuple[int,int]) -> Optional[pd.DataFrame]:
    try:
        fp = _recon_cache_fp(key)
        if not fp.exists():
            return None
        # TTL check against file mtime
        ttl = int(os.environ.get('RECON_CACHE_TTL_SEC', '1800'))
        age = max(0, int(time.time() - fp.stat().st_mtime))
        if age > max(0, ttl):
            return None
        df = pd.read_csv(fp)
        return df
    except Exception:
        return None


def _recon_cache_save_to_disk(key: tuple[int,int], df: pd.DataFrame) -> None:
    try:
        fp = _recon_cache_fp(key)
        df.to_csv(fp, index=False)
    except Exception:
        pass


def _find_latest_props_cache(prefer_season: Optional[int] = None, prefer_week: Optional[int] = None) -> Optional[tuple[Path, int, int]]:
    """Find the most suitable player props cache file on disk.

    Preference order:
    - Exact match for prefer_season and prefer_week
    - Same season, highest available week
    - Any season, latest by (season, week) descending
    Returns (path, season, week) or None if none found.
    """
    try:
        pat = re.compile(r"^player_props_(\d{4})_wk(\d{1,2})\.csv$")
        candidates: list[tuple[int,int,Path]] = []
        for p in DATA_DIR.glob('player_props_*_wk*.csv'):
            m = pat.match(p.name)
            if not m:
                continue
            try:
                s = int(m.group(1)); w = int(m.group(2))
                candidates.append((s, w, p))
            except Exception:
                continue
        if not candidates:
            return None
        # Exact match
        if prefer_season is not None and prefer_week is not None:
            for s, w, p in candidates:
                if s == prefer_season and w == prefer_week:
                    return p, s, w
        # Same season, highest week
        if prefer_season is not None:
            same = [(s, w, p) for s, w, p in candidates if s == prefer_season]
            if same:
                s, w, p = max(same, key=lambda t: t[1])
                return p, s, w
        # Global latest
        s, w, p = max(candidates, key=lambda t: (t[0], t[1]))
        return p, s, w
    except Exception:
        return None


def _load_current_week_override() -> Optional[tuple[int, int]]:
    """Return (season, week) if override is configured via env or data file.

    Priority:
      1) Env: CURRENT_SEASON and CURRENT_WEEK (or DEFAULT_SEASON/DEFAULT_WEEK)
      2) File: nfl_compare/data/current_week.json with {"season": YYYY, "week": N}
    """
    try:
        # Env-based
        def _get_int(name: str) -> Optional[int]:
            v = os.environ.get(name)
            if v is None:
                return None
            try:
                return int(str(v).strip())
            except Exception:
                return None
        season_env = _get_int('CURRENT_SEASON') or _get_int('DEFAULT_SEASON')
        week_env = _get_int('CURRENT_WEEK') or _get_int('DEFAULT_WEEK')
        if season_env and week_env:
            return int(season_env), int(week_env)
        # File-based
        fp = DATA_DIR / 'current_week.json'
        if fp.exists():
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                s = int(obj.get('season')) if obj.get('season') is not None else None
                w = int(obj.get('week')) if obj.get('week') is not None else None
                if s and w:
                    return s, w
            except Exception:
                return None
    except Exception:
        return None
    return None


def _admin_auth_ok(req) -> bool:
    key = os.environ.get('ADMIN_KEY') or os.environ.get('ADMIN_TOKEN')
    if not key:
        return False
    sent = req.args.get('key') or req.headers.get('X-Admin-Key') or req.headers.get('Authorization', '').replace('Bearer ', '')
    return bool(sent) and (sent == key)


def _append_log(line: str) -> None:
    try:
        ts = datetime.utcnow().isoformat(timespec='seconds')
        msg = f"[{ts}] {line.rstrip()}"
        _job_state['logs'].append(msg)
        # cap size
        if len(_job_state['logs']) > 1000:
            del _job_state['logs'][:-500]
        # also mirror into the run's file log if available so UI tail shows admin messages
        try:
            lf = _job_state.get('log_file')
            if lf:
                with open(lf, 'a', encoding='utf-8', errors='ignore') as f:
                    f.write(msg + "\n")
        except Exception:
            pass
    except Exception:
        pass

def _ensure_logs_dir() -> Path:
    p = BASE_DIR / 'logs'
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


def _tail_file(fp: Path, max_lines: int = 200) -> list[str]:
    try:
        if not fp.exists():
            return []
        # Simple tail: read and slice last N lines; files are small
        with fp.open('r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        return [ln.rstrip('\n') for ln in lines[-max_lines:]]
    except Exception:
        return []


def _run_to_file(cmd: list[str] | str, log_fp: Path, cwd: Path | None = None, env: dict | None = None) -> int:
    if isinstance(cmd, list):
        popen_cmd = cmd
    else:
        popen_cmd = shlex.split(cmd)
    # Open file for append to capture output without blocking the web worker
    with log_fp.open('a', encoding='utf-8', errors='ignore') as out:
        out.write(f"[{datetime.utcnow().isoformat(timespec='seconds')}] Starting: {' '.join(popen_cmd)}\n")
        out.flush()
        proc = subprocess.Popen(
            popen_cmd,
            cwd=str(cwd) if cwd else None,
            env={**os.environ, **(env or {})},
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        proc.wait()
        out.write(f"[{datetime.utcnow().isoformat(timespec='seconds')}] Exited with code {proc.returncode}\n")
        out.flush()
        return int(proc.returncode or 0)


def _git_commit_and_push(commit_message: str) -> tuple[bool, str]:
    """Commit any changes and push to origin using a token if provided.
    Requires env ADMIN_KEY for route access and optionally GITHUB_TOKEN or GH_PAT.
    """
    try:
        # Load secrets from files if available (supports *_FILE and common mounts)
        try:
            from nfl_compare.src.config import load_env as _load_env  # lazy import
            _load_env()
        except Exception:
            pass
        # Ensure git user
        _ = subprocess.run(['git', 'config', 'user.email'], capture_output=True, text=True)
        if _.returncode != 0 or not _.stdout.strip():
            subprocess.run(['git', 'config', 'user.email', os.environ.get('GIT_AUTHOR_EMAIL', 'render-bot@example.com')], check=False)
        _ = subprocess.run(['git', 'config', 'user.name'], capture_output=True, text=True)
        if _.returncode != 0 or not _.stdout.strip():
            subprocess.run(['git', 'config', 'user.name', os.environ.get('GIT_AUTHOR_NAME', 'Render Bot')], check=False)

        # Stage changes safely: update tracked files and add new allowed data artifacts only
        # 1) Update tracked files
        _ = subprocess.run(['git', 'add', '-u'], capture_output=True, text=True)
        if _.returncode != 0:
            return False, f"git add -u failed: {_.stderr.strip()}"
        # 2) Add new allowed files under nfl_compare/data with safe extensions
        try:
            from pathlib import Path as _P
            data_dir = _P('nfl_compare') / 'data'
            exts = {'.csv', '.json', '.parquet'}
            to_add = []
            if data_dir.exists():
                for p in data_dir.rglob('*'):
                    if p.is_file() and p.suffix.lower() in exts:
                        to_add.append(str(p))
            if to_add:
                # Add in batches to avoid command length limits
                batch = []
                for fp in to_add:
                    batch.append(fp)
                    if len(batch) >= 200:
                        subprocess.run(['git', 'add'] + batch, check=False)
                        batch = []
                if batch:
                    subprocess.run(['git', 'add'] + batch, check=False)
        except Exception:
            pass

        # 3) Explicitly unstage suspicious files (potential secrets)
        try:
            staged = subprocess.run(['git', 'diff', '--cached', '--name-only'], capture_output=True, text=True)
            files = (staged.stdout or '').splitlines()
            suspicious = []
            block_names = {
                'GITHUB_TOKEN', 'GH_PAT', 'RENDER_GIT_TOKEN', 'ODDS_API_KEY', '.env', '.env.local', '.env.prod'
            }
            for fn in files:
                base = os.path.basename(fn)
                if base in block_names or 'secret' in fn.lower() or 'secrets' in fn.lower():
                    suspicious.append(fn)
            for fn in suspicious:
                subprocess.run(['git', 'restore', '--staged', fn], check=False)
            if suspicious:
                _append_log(f"Skipped staging suspicious files: {', '.join(suspicious)}")
        except Exception:
            pass

        # Skip empty commit
        diff = subprocess.run(['git', 'diff', '--cached', '--quiet'], capture_output=True)
        if diff.returncode == 0:
            return True, 'No changes to commit.'

        cm = subprocess.run(['git', 'commit', '-m', commit_message], capture_output=True, text=True)
        if cm.returncode != 0:
            return False, f"git commit failed: {cm.stderr.strip()}"

        # Determine push URL
        # Prefer explicit GIT_REMOTE_URL; otherwise derive from origin
        push_url = os.environ.get('GIT_REMOTE_URL', '').strip()
        if not push_url:
            ru = subprocess.run(['git', 'remote', 'get-url', 'origin'], capture_output=True, text=True)
            if ru.returncode != 0:
                return False, f"git remote get-url failed: {ru.stderr.strip()}"
            push_url = ru.stdout.strip()

        token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_PAT') or os.environ.get('RENDER_GIT_TOKEN')
        if token and push_url.startswith('https://') and 'github.com' in push_url:
            # Insert token; prefer x-access-token for GitHub Apps tokens; PAT also works
            push_url_auth = push_url.replace('https://', f"https://x-access-token:{token}@")
        else:
            # If no token and URL is GitHub HTTPS without embedded creds, return a clear error
            if (not token) and push_url.startswith('https://') and 'github.com' in push_url and ('@' not in push_url):
                return False, (
                    "No GitHub token configured. Set GITHUB_TOKEN (or GH_PAT / RENDER_GIT_TOKEN) "
                    "and optionally GIT_REMOTE_URL to your repo HTTPS URL."
                )
            push_url_auth = push_url

        ps = subprocess.run(['git', 'push', push_url_auth, 'HEAD:master'], capture_output=True, text=True)
        if ps.returncode != 0:
            err = (ps.stderr or ps.stdout).strip()
            # Attempt auto-cleanup if GitHub Push Protection blocks secrets (GH013)
            if 'GH013' in err and 'Push cannot contain secrets' in err:
                try:
                    # Identify suspicious files in the last commit
                    show = subprocess.run(['git', 'show', '--name-only', '--pretty=', 'HEAD'], capture_output=True, text=True)
                    files_last = (show.stdout or '').splitlines()
                    block_names = {
                        'GITHUB_TOKEN', 'GH_PAT', 'RENDER_GIT_TOKEN', 'ODDS_API_KEY', '.env', '.env.local', '.env.prod'
                    }
                    sus = []
                    for fn in files_last:
                        base = os.path.basename(fn)
                        if base in block_names or 'secret' in fn.lower() or 'secrets' in fn.lower():
                            sus.append(fn)
                    if sus:
                        # Undo the last commit but keep changes staged
                        subprocess.run(['git', 'reset', '--soft', 'HEAD~1'], check=False)
                        # Unstage and leave suspicious files out of the commit
                        for fn in sus:
                            subprocess.run(['git', 'restore', '--staged', fn], check=False)
                        # Recommit without the suspicious files
                        cm2 = subprocess.run(['git', 'commit', '-m', f"{commit_message} [redacted secrets]"], capture_output=True, text=True)
                        if cm2.returncode != 0:
                            return False, f"git commit (after redaction) failed: {cm2.stderr.strip()}"
                        # Retry push
                        ps2 = subprocess.run(['git', 'push', push_url_auth, 'HEAD:master'], capture_output=True, text=True)
                        if ps2.returncode == 0:
                            return True, 'Pushed successfully after redacting suspicious files.'
                        err2 = (ps2.stderr or ps2.stdout).strip()
                        return False, f"git push failed after redaction: {err2}"
                except Exception as e:  # noqa: BLE001
                    _log_once("predictions-exc", f"Failed loading predictions: {e}")
                    # Fall through to empty frame
                    pass
    except Exception as e:
        return False, f"git push exception: {e}"


def _daily_update_job(do_push: bool) -> None:
    _job_state['running'] = True
    _job_state['started_at'] = datetime.utcnow().isoformat()
    _job_state['ended_at'] = None
    _job_state['ok'] = None
    _job_state['logs'] = []
    # Prepare log file for this run
    logs_dir = _ensure_logs_dir()
    stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"web_daily_update_{stamp}.log"
    _job_state['log_file'] = str(log_file)
    try:
        _append_log('Starting daily update...')
        # Prefer light behavior on Render
        env = {'RENDER': os.environ.get('RENDER', 'true')}
        rc = _run_to_file([sys.executable, '-m', 'nfl_compare.src.daily_updater'], log_file, cwd=BASE_DIR, env=env)
        _append_log(f'daily_updater exited with code {rc}')
        ok = (rc == 0)
        if ok and do_push:
            _append_log('Committing and pushing changes to Git...')
            ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')
            ok2, msg = _git_commit_and_push(f'chore(data): web-triggered daily update {ts}')
            _append_log(msg)
            ok = ok and ok2
        elif ok and not do_push:
            _append_log('Push disabled by request; skipping git commit/push.')
        _job_state['ok'] = ok
    except Exception as e:
        _append_log(f'Exception: {e}')
        _job_state['ok'] = False
    finally:
        _job_state['running'] = False
        _job_state['ended_at'] = datetime.utcnow().isoformat()


@app.route('/api/admin/daily-update', methods=['POST','GET'])
def api_admin_daily_update():
    if not _admin_auth_ok(request):
        return jsonify({'error': 'unauthorized'}), 401
    if _job_state['running']:
        return jsonify({'status': 'already-running', 'started_at': _job_state['started_at']}), 409
    do_push = (request.args.get('push', '1') in ('1','true','yes'))
    t = threading.Thread(target=_daily_update_job, args=(do_push,), daemon=True)
    t.start()
    return jsonify({'status': 'started', 'push': do_push, 'started_at': datetime.utcnow().isoformat()}), 202


@app.route('/api/admin/daily-update/status', methods=['GET'])
def api_admin_daily_status():
    if not _admin_auth_ok(request):
        return jsonify({'error': 'unauthorized'}), 401
    tail = int(request.args.get('tail', '200'))
    # Combine lightweight web logs with tail of the updater file
    log_lines = list(_job_state['logs'][-tail:])
    try:
        lf = _job_state.get('log_file')
        if lf:
            from pathlib import Path as _P
            file_tail = _tail_file(_P(lf), tail)
            if file_tail:
                # Merge: file tail first (from updater), then any recent admin messages
                merged = file_tail + [ln for ln in log_lines if ln not in file_tail]
                # Keep only the last N lines to avoid growth
                log_lines = merged[-tail:]
    except Exception:
        pass
    return jsonify({
        'running': _job_state['running'],
        'started_at': _job_state['started_at'],
        'ended_at': _job_state['ended_at'],
        'ok': _job_state['ok'],
        'logs': log_lines,
        'log_file': _job_state.get('log_file'),
    })


@app.route('/api/admin/ping', methods=['GET'])
def api_admin_ping():
    if not _admin_auth_ok(request):
        return jsonify({'error': 'unauthorized'}), 401
    return jsonify({'status': 'ok'}), 200


@app.route('/api/admin/recon-cache/clear', methods=['POST','GET'])
def api_admin_clear_recon_cache():
    if not _admin_auth_ok(request):
        return jsonify({'error': 'unauthorized'}), 401
    try:
        n = len(_recon_cache)
        _recon_cache.clear()
        return jsonify({'status': 'cleared', 'entries': n})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



def _load_predictions() -> pd.DataFrame:
    """Load predictions.csv if present; return empty DataFrame if missing."""
    try:
        dfs = []
        # Load each source with a tag so we can prioritize later
        if PRED_FILE.exists():
            try:
                d0 = pd.read_csv(PRED_FILE)
                d0['pred_source'] = 'pred'
                dfs.append(d0)
            except Exception as e:
                _log_once('predictions-read-fail', f'predictions.csv read error: {e}')
        if PRED_WEEK_FILE.exists():
            try:
                d1 = pd.read_csv(PRED_WEEK_FILE)
                d1['pred_source'] = 'week'
                dfs.append(d1)
            except Exception as e:
                _log_once('predictions-week-read-fail', f'predictions_week.csv read error: {e}')
        if LOCKED_PRED_FILE.exists():
            try:
                d2 = pd.read_csv(LOCKED_PRED_FILE)
                d2['pred_source'] = 'locked'
                dfs.append(d2)
            except Exception as e:
                _log_once('predictions-locked-read-fail', f'predictions_locked.csv read error: {e}')
        synth_file = DATA_DIR / 'predictions_synth.csv'
        if synth_file.exists():
            try:
                d3 = pd.read_csv(synth_file)
                d3['pred_source'] = 'synth'
                dfs.append(d3)
            except Exception:
                _log_once('predictions-synth-load-fail', 'Failed loading predictions_synth.csv')

        # Fallback: check top-level ./data directory for predictions if nfl_compare/data is empty
        if not dfs:
            try:
                alt_dir = BASE_DIR / 'data'
                alt_files = {
                    'pred': alt_dir / 'predictions.csv',
                    'week': alt_dir / 'predictions_week.csv',
                    'locked': alt_dir / 'predictions_locked.csv',
                    'synth': alt_dir / 'predictions_synth.csv',
                }
                for tag, path in alt_files.items():
                    if path.exists():
                        try:
                            d = pd.read_csv(path)
                            d['pred_source'] = tag
                            dfs.append(d)
                        except Exception:
                            _log_once(f'predictions-{tag}-alt-read-fail', f'Failed reading {path}')
            except Exception:
                pass

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            # Normalize typical columns if present
            for c in ("week", "season"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            # Prefer rows with finals and week-level source when deduplicating by game_id
            if 'game_id' in df.columns:
                # Compute helper columns in a single assign to avoid fragmentation
                try:
                    if {'home_score','away_score'}.issubset(df.columns):
                        has_finals = df['home_score'].notna() & df['away_score'].notna()
                    else:
                        has_finals = pd.Series(False, index=df.index)
                except Exception:
                    has_finals = pd.Series(False, index=df.index)
                # Always prefer locked snapshots over any other source to maintain credibility post-game
                # Higher is better
                src_rank = {'locked': 100, 'week': 3, 'pred': 2, 'synth': 0}
                pred_source = df['pred_source'] if 'pred_source' in df.columns else pd.Series(None, index=df.index)
                src_priority = pred_source.map(lambda s: src_rank.get(str(s), -1))
                is_locked = (pred_source == 'locked')
                df = df.assign(has_finals=has_finals, src_priority=src_priority, is_locked=is_locked)
                # Sort by is_locked first (desc), then finals, then source priority; keep first per game_id
                df = df.sort_values(by=['is_locked','has_finals','src_priority'], ascending=[False, False, False])
                df = df.drop_duplicates(subset=['game_id'], keep='first')
                # Clean helper columns
                df = df.drop(columns=[c for c in ['has_finals','src_priority','is_locked'] if c in df.columns])
            # Normalize typical columns if present
            for c in ("week", "season"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            # Sort by season/week/date if available
            sort_cols = [c for c in ["season", "week", "game_date"] if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols)
            return df
        else:
            _log_once("predictions-missing", f"Predictions sources missing or empty under {DATA_DIR}")
    except Exception as e:  # noqa: BLE001
        _log_once("predictions-exc", f"Failed loading predictions: {e}")
        # Fall through to empty frame
        pass
    return pd.DataFrame()


def _load_games() -> pd.DataFrame:
    """Load games.csv plus union of games_normalized.csv (if present).

    The normalized file extends sparse schedules (e.g., for early weeks) using locked predictions.
    We avoid duplicate game_id rows by preferring the original games.csv entries.
    """
    try:
        fp = DATA_DIR / "games.csv"
        norm_fp = DATA_DIR / "games_normalized.csv"
        df = pd.DataFrame()
        if fp.exists():
            try:
                df = pd.read_csv(fp)
            except Exception as e:  # noqa: BLE001
                _log_once("games-read-fail", f"Failed reading games.csv: {e}")
        else:
            _log_once("games-missing", f"games.csv not found under {DATA_DIR}")

        if not df.empty:
            for c in ("week", "season"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

        # Union normalized games for additional coverage
        if norm_fp.exists():
            try:
                nf = pd.read_csv(norm_fp)
                for c in ("week","season"):
                    if c in nf.columns:
                        nf[c] = pd.to_numeric(nf[c], errors='coerce')
                if 'game_id' in nf.columns:
                    if df.empty or 'game_id' not in df.columns:
                        df = nf
                    else:
                        base_ids = set(df['game_id'].dropna().astype(str))
                        add = nf[~nf['game_id'].astype(str).isin(base_ids)].copy()
                        if not add.empty:
                            df = pd.concat([df, add], ignore_index=True)
            except Exception as e:  # noqa: BLE001
                _log_once("games-norm-read-fail", f"Failed reading games_normalized.csv: {e}")

        # Fallback: read from top-level ./data if nfl_compare/data was empty
        if df.empty:
            alt_dir = BASE_DIR / 'data'
            alt_fp = alt_dir / 'games.csv'
            alt_norm = alt_dir / 'games_normalized.csv'
            if alt_fp.exists():
                try:
                    df = pd.read_csv(alt_fp)
                    for c in ("week","season"):
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors='coerce')
                except Exception as e:  # noqa: BLE001
                    _log_once("games-alt-read-fail", f"Failed reading {alt_fp}: {e}")
            # Union alt normalized
            if alt_norm.exists():
                try:
                    nf = pd.read_csv(alt_norm)
                    for c in ("week","season"):
                        if c in nf.columns:
                            nf[c] = pd.to_numeric(nf[c], errors='coerce')
                    if 'game_id' in nf.columns:
                        if df.empty or 'game_id' not in df.columns:
                            df = nf
                        else:
                            base_ids = set(df['game_id'].dropna().astype(str))
                            add = nf[~nf['game_id'].astype(str).isin(base_ids)].copy()
                            if not add.empty:
                                df = pd.concat([df, add], ignore_index=True)
                except Exception as e:  # noqa: BLE001
                    _log_once("games-alt-norm-read-fail", f"Failed reading {alt_norm}: {e}")

        if not df.empty:
            sort_cols = [c for c in ["season", "week", "game_date", "date"] if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols)
        return df
    except Exception as e:  # noqa: BLE001
        _log_once("games-exc", f"Failed loading games (union): {e}")
        pass
    return pd.DataFrame()


@app.route('/api/health/data')
def api_health_data():
    """Lightweight data presence/status report for debugging deployments."""
    try:
        status = {}
        core_files = {
            'predictions.csv': PRED_FILE,
            'predictions_week.csv': PRED_WEEK_FILE,
            'predictions_locked.csv': LOCKED_PRED_FILE,
            'games.csv': DATA_DIR / 'games.csv',
            'lines.csv': DATA_DIR / 'lines.csv',
            'nfl_team_assets.json': ASSETS_FILE,
        }
        for name, path in core_files.items():
            info = {'exists': path.exists()}
            try:
                if path.exists() and path.is_file():
                    info['size_bytes'] = path.stat().st_size
                    if name.endswith('.csv'):
                        # Read only header + first row for speed
                        import csv as _csv
                        with open(path, 'r', encoding='utf-8') as f:
                            rdr = _csv.reader(f)
                            header = next(rdr, [])
                            first = next(rdr, None)
                        info['columns'] = header
                        info['has_rows'] = first is not None
                status[name] = info
            except Exception as e:  # noqa: BLE001
                _log_once("games-exc", f"Failed loading games.csv: {e}")
                pass
        # Attempt quick counts (avoid full load if large)
        games_df = _load_games()
        preds_df = _load_predictions()
        status['counts'] = {
            'games_rows': 0 if games_df is None else len(games_df),
            'predictions_rows': 0 if preds_df is None else len(preds_df)
        }
        status['data_dir'] = str(DATA_DIR)
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/props/teams')
def api_props_teams():
    """Return the list of teams with games for the selected season/week.

    Query params:
      - season (int, optional)
      - week (int, optional)
    If missing, defaults to most recent season and inferred current week from games.csv when possible.
    """
    try:
        games_df = _load_games()
        season = request.args.get('season')
        week = request.args.get('week')
        season_i: Optional[int] = None
        week_i: Optional[int] = None
        try:
            season_i = int(season) if season else None
        except Exception:
            season_i = None
        try:
            week_i = int(week) if week else None
        except Exception:
            week_i = None

        # Default season/week using current-week inference (respects overrides via current_week.json / env)
        if (season_i is None) or (week_i is None):
            try:
                inferred = _infer_current_season_week(games_df) if (games_df is not None and not games_df.empty) else None
                if inferred is not None:
                    if season_i is None:
                        season_i = int(inferred[0])
                    if week_i is None:
                        week_i = int(inferred[1])
            except Exception:
                pass
        # Fallbacks if inference failed
        if games_df is not None and not games_df.empty:
            if season_i is None and 'season' in games_df.columns and not games_df['season'].isna().all():
                try:
                    season_i = int(pd.to_numeric(games_df['season'], errors='coerce').dropna().max())
                except Exception:
                    season_i = None
            if week_i is None and 'week' in games_df.columns:
                try:
                    if season_i is not None:
                        week_i = int(pd.to_numeric(games_df.loc[games_df['season'] == season_i, 'week'], errors='coerce').dropna().max())
                    else:
                        week_i = int(pd.to_numeric(games_df['week'], errors='coerce').dropna().max())
                except Exception:
                    week_i = 1

        # Helper to extract team list from a DataFrame with home_team/away_team
        def _teams_from_match_df(df: pd.DataFrame) -> list[str]:
            try:
                if df is None or df.empty:
                    return []
                cols = [c for c in ['home_team','away_team'] if c in df.columns]
                if not cols:
                    return []
                vals: list[str] = []
                for c in cols:
                    vals.extend(df[c].astype(str).tolist())
                return sorted(sorted(set(t for t in vals if isinstance(t, str) and t.strip())))
            except Exception:
                return []

        # Primary: teams from games.csv for the selected season/week
        teams: list[str] = []
        view = games_df.copy() if (games_df is not None and not games_df.empty) else pd.DataFrame()
        if season_i is not None and 'season' in view.columns:
            view = view[view['season'] == season_i]
        if week_i is not None and 'week' in view.columns:
            view = view[view['week'] == week_i]
        teams = _teams_from_match_df(view)

        # Fallback A: if no teams for that week, try betting lines for that week
        if not teams:
            try:
                from nfl_compare.src.data_sources import load_lines as _load_lines_for_teams
            except Exception:
                _load_lines_for_teams = None
            if _load_lines_for_teams is not None:
                try:
                    ldf = _load_lines_for_teams()
                except Exception:
                    ldf = pd.DataFrame()
                if ldf is not None and not ldf.empty:
                    lview = ldf.copy()
                    if season_i is not None and 'season' in lview.columns:
                        lview = lview[lview['season'] == season_i]
                    if week_i is not None and 'week' in lview.columns:
                        lview = lview[lview['week'] == week_i]
                    teams = _teams_from_match_df(lview)

        # Fallback B: if still empty, broaden to all teams seen in the season (games.csv then lines.csv)
        if not teams:
            season_teams: list[str] = []
            if games_df is not None and not games_df.empty and season_i is not None:
                gseason = games_df[games_df['season'] == season_i]
                season_teams = _teams_from_match_df(gseason)
            if not season_teams:
                try:
                    from nfl_compare.src.data_sources import load_lines as _load_lines_for_teams2
                except Exception:
                    _load_lines_for_teams2 = None
                if _load_lines_for_teams2 is not None:
                    try:
                        ldf2 = _load_lines_for_teams2()
                    except Exception:
                        ldf2 = pd.DataFrame()
                    if ldf2 is not None and not ldf2.empty and season_i is not None:
                        season_teams = _teams_from_match_df(ldf2[ldf2['season'] == season_i])
            if season_teams:
                teams = season_teams

        # Fallback C: precomputed props cache for that week, if available
        if not teams:
            try:
                if season_i is not None and week_i is not None:
                    cache_fp = DATA_DIR / f"player_props_{int(season_i)}_wk{int(week_i)}.csv"
                    if cache_fp.exists():
                        try:
                            dfp = pd.read_csv(cache_fp)
                            if dfp is not None and not dfp.empty and 'team' in dfp.columns:
                                vals = dfp['team'].astype(str).tolist()
                                teams = sorted(sorted(set(t for t in vals if isinstance(t, str) and t.strip())))
                        except Exception:
                            pass
            except Exception:
                pass

        # Fallback D: all NFL team abbreviations from assets file
        if not teams:
            try:
                if ASSETS_FILE.exists():
                    with open(ASSETS_FILE, 'r', encoding='utf-8') as f:
                        assets = json.load(f)  # {FullName: {abbr: "..."}}
                    abbrs = []
                    for k, v in (assets or {}).items():
                        ab = v.get('abbr') if isinstance(v, dict) else None
                        if ab:
                            abbrs.append(str(ab))
                    teams = sorted(sorted(set(abbrs)))
            except Exception:
                pass

        return jsonify({'season': season_i, 'week': week_i, 'teams': teams})
    except Exception as e:
        return jsonify({'error': f'teams endpoint failed: {e}'}), 500


def _build_week_view(pred_df: pd.DataFrame, games_df: pd.DataFrame, season: Optional[int], week: Optional[int]) -> pd.DataFrame:
    """Return the base week view (games + attached model predictions).

    Responsibilities:
    1. Choose a base set of rows for (season, week): prefer games.csv; else synthesize from lines; else from predictions.
    2. Attach prediction columns without overwriting core game fields.
    3. Do NOT derive market-based synthetic prediction values here (handled later in a dedicated helper).
    """
    # Helper to filter by season/week defensively
    def _filter_sw(df: pd.DataFrame) -> pd.DataFrame:
        out = df
        if season is not None and 'season' in out.columns:
            out = out[out['season'] == season]
        if week is not None and 'week' in out.columns:
            out = out[out['week'] == week]
        return out

    # 1. Establish base view from games if possible
    view = pd.DataFrame()
    if games_df is not None and not games_df.empty:
        try:
            view = _filter_sw(games_df.copy())
        except Exception:
            view = pd.DataFrame()

    # 1b. If games missing, synthesize from lines
    if view.empty:
        try:
            from nfl_compare.src.data_sources import load_lines as _load_lines_for_view
            lines_all = _load_lines_for_view()
        except Exception:
            lines_all = None
        if lines_all is not None and not getattr(lines_all, 'empty', True):
            try:
                l = _filter_sw(lines_all.copy())
                keep_cols = [c for c in ['season','week','game_id','game_date','date','home_team','away_team'] if c in l.columns]
                if keep_cols and not l.empty:
                    synth = l[keep_cols].drop_duplicates()
                    if 'game_date' not in synth.columns and 'date' in synth.columns:
                        synth = synth.rename(columns={'date': 'game_date'})
                    for c in ['season','week','home_team','away_team']:
                        if c not in synth.columns:
                            synth[c] = None
                    view = synth
            except Exception:
                pass

    # 1c. Fallback to predictions scaffolding
    if view.empty and pred_df is not None and not pred_df.empty:
        try:
            pf = _filter_sw(pred_df.copy())
            keep_cols = [c for c in ['season','week','game_id','game_date','date','home_team','away_team'] if c in pf.columns]
            if keep_cols and not pf.empty:
                core = pf[keep_cols].drop_duplicates()
                if 'game_date' not in core.columns and 'date' in core.columns:
                    core = core.rename(columns={'date': 'game_date'})
                for c in ['season','week','home_team','away_team']:
                    if c not in core.columns:
                        core[c] = None
                view = core
        except Exception:
            pass

    # If still empty, nothing else to do
    if view.empty:
        # As a last resort return filtered predictions (raw) for transparency
        if pred_df is not None and not pred_df.empty:
            return _filter_sw(pred_df.copy())
        return view

    # 2. Attach predictions (if any) without overwriting core keys
    if pred_df is None or pred_df.empty:
        return view

    p = pred_df.copy()
    try:
        from nfl_compare.src.team_normalizer import normalize_team_name as _norm_team
        for col in ('home_team','away_team'):
            if col in p.columns:
                p[col] = p[col].astype(str).apply(_norm_team)
        for col in ('home_team','away_team'):
            if col in view.columns:
                view[col] = view[col].astype(str).apply(_norm_team)
    except Exception:
        pass

    # Do NOT carry market/odds fields from predictions.csv; they'll be joined later from lines.csv.
    core_keys = {
        'season','week','game_id','game_date','date','home_team','away_team','home_score','away_score',
        # Odds/market fields we explicitly exclude from pred merge to avoid stale leakage
        'spread_home','open_spread_home','close_spread_home',
        'total','open_total','close_total',
        'moneyline_home','moneyline_away',
        'spread_home_price','spread_away_price','total_over_price','total_under_price',
        'market_spread_home','market_total'
    }
    pred_cols = [c for c in p.columns if c not in core_keys]
    merged = view.copy()

    # Merge by game_id first
    if 'game_id' in merged.columns and 'game_id' in p.columns and not p['game_id'].isna().all():
        try:
            right_gid = p[['game_id'] + [c for c in pred_cols if c in p.columns]].drop_duplicates()
            merged = merged.merge(right_gid, on='game_id', how='left')
        except Exception:
            pass

    # Secondary merge by (season, week, home, away) to fill gaps
    team_keys = [c for c in ['season','week','home_team','away_team'] if c in merged.columns and c in p.columns]
    if len(team_keys) == 4:
        try:
            right_team = p[team_keys + pred_cols].drop_duplicates()
            merged_tw = merged.merge(right_team, on=team_keys, how='left', suffixes=('', '_p2'))
            for c in pred_cols:
                if c in merged_tw.columns and f'{c}_p2' in merged_tw.columns:
                    merged_tw[c] = merged_tw[c].where(merged_tw[c].notna(), merged_tw[f'{c}_p2'])
            drop2 = [c for c in merged_tw.columns if c.endswith('_p2')]
            if drop2:
                merged_tw = merged_tw.drop(columns=drop2)
            merged = merged_tw
        except Exception:
            pass

    # 2c. Coalesce final scores/status from predictions_week/locked where games rows are missing them
    try:
        if not merged.empty and pred_df is not None and not pred_df.empty:
            p = pred_df.copy()
            # Only proceed if we have game_id linkage and any score columns
            if ('game_id' in merged.columns) and ('game_id' in p.columns) and (('home_score' in p.columns) or ('away_score' in p.columns) or ('status' in p.columns)):
                keep = ['game_id']
                if 'home_score' in p.columns:
                    keep.append('home_score')
                if 'away_score' in p.columns:
                    keep.append('away_score')
                if 'status' in p.columns:
                    keep.append('status')
                p_scores = p[keep].drop_duplicates()
                # Avoid overwriting: merge with suffixes and fill only where base is null
                merged_sc = merged.merge(p_scores, on='game_id', how='left', suffixes=('', '_p'))
                # Coerce numeric for safe null checks
                for c in ('home_score','away_score'):
                    if c in merged_sc.columns:
                        merged_sc[c] = pd.to_numeric(merged_sc[c], errors='coerce')
                    if f"{c}_p" in merged_sc.columns:
                        merged_sc[f"{c}_p"] = pd.to_numeric(merged_sc[f"{c}_p"], errors='coerce')
                # Ensure score/status columns exist; then fill base only when missing
                if 'home_score_p' in merged_sc.columns:
                    if 'home_score' not in merged_sc.columns:
                        merged_sc['home_score'] = merged_sc['home_score_p']
                    else:
                        merged_sc['home_score'] = merged_sc['home_score'].where(merged_sc['home_score'].notna(), merged_sc['home_score_p'])
                if 'away_score_p' in merged_sc.columns:
                    if 'away_score' not in merged_sc.columns:
                        merged_sc['away_score'] = merged_sc['away_score_p']
                    else:
                        merged_sc['away_score'] = merged_sc['away_score'].where(merged_sc['away_score'].notna(), merged_sc['away_score_p'])
                # Status: set if missing, or create if absent
                if 'status_p' in merged_sc.columns:
                    if 'status' not in merged_sc.columns:
                        merged_sc['status'] = merged_sc['status_p']
                    else:
                        merged_sc['status'] = merged_sc['status'].where(merged_sc['status'].notna(), merged_sc['status_p'])
                # Drop helper columns
                drop_sc = [c for c in merged_sc.columns if c.endswith('_p') and c in ('home_score_p','away_score_p','status_p')]
                if drop_sc:
                    merged_sc = merged_sc.drop(columns=drop_sc)
                merged = merged_sc
                # If some rows still lack scores/status (possible when game_id mismatches), try a secondary fill by team keys
                try:
                    need_scores = None
                    if {'home_score','away_score'}.issubset(merged.columns):
                        need_scores = merged['home_score'].isna() | merged['away_score'].isna()
                    elif ('home_score' in merged.columns) or ('away_score' in merged.columns):
                        # If only one exists, still consider missing where present is NaN
                        cols_present = [c for c in ('home_score','away_score') if c in merged.columns]
                        if cols_present:
                            need_scores = merged[cols_present].isna().any(axis=1)
                    # Only proceed if we have team keys on both frames and some rows still need fill
                    team_keys = [c for c in ['season','week','home_team','away_team'] if c in merged.columns and c in p.columns]
                    if need_scores is not None and need_scores.any() and len(team_keys) == 4:
                        keep2 = team_keys.copy()
                        for c in ('home_score','away_score','status'):
                            if c in p.columns and c not in keep2:
                                keep2.append(c)
                        p_sw = p[keep2].drop_duplicates()
                        merged_sw = merged.merge(p_sw, on=team_keys, how='left', suffixes=('', '_pt'))
                        # Coerce numeric for safe null checks
                        for c in ('home_score','away_score'):
                            if c in merged_sw.columns:
                                merged_sw[c] = pd.to_numeric(merged_sw[c], errors='coerce')
                            if f"{c}_pt" in merged_sw.columns:
                                merged_sw[f"{c}_pt"] = pd.to_numeric(merged_sw[f"{c}_pt"], errors='coerce')
                        # Fill only where missing
                        if 'home_score_pt' in merged_sw.columns:
                            if 'home_score' not in merged_sw.columns:
                                merged_sw['home_score'] = merged_sw['home_score_pt']
                            else:
                                merged_sw['home_score'] = merged_sw['home_score'].where(merged_sw['home_score'].notna(), merged_sw['home_score_pt'])
                        if 'away_score_pt' in merged_sw.columns:
                            if 'away_score' not in merged_sw.columns:
                                merged_sw['away_score'] = merged_sw['away_score_pt']
                            else:
                                merged_sw['away_score'] = merged_sw['away_score'].where(merged_sw['away_score'].notna(), merged_sw['away_score_pt'])
                        if 'status_pt' in merged_sw.columns:
                            if 'status' not in merged_sw.columns:
                                merged_sw['status'] = merged_sw['status_pt']
                            else:
                                merged_sw['status'] = merged_sw['status'].where(merged_sw['status'].notna(), merged_sw['status_pt'])
                        drop_pt = [c for c in merged_sw.columns if c.endswith('_pt') and c in ('home_score_pt','away_score_pt','status_pt')]
                        if drop_pt:
                            merged_sw = merged_sw.drop(columns=drop_pt)
                        merged = merged_sw
                except Exception:
                    pass
                # 2d. For finalized games, backfill closing lines and moneylines from predictions
                try:
                    finals_mask = None
                    if {'home_score','away_score'}.issubset(merged.columns):
                        finals_mask = pd.to_numeric(merged['home_score'], errors='coerce').notna() & pd.to_numeric(merged['away_score'], errors='coerce').notna()
                    if finals_mask is not None and finals_mask.any():
                        # Select odds/close fields from predictions to backfill
                        odds_cols = [
                            'close_spread_home','close_total',
                            'moneyline_home','moneyline_away',
                            'spread_home','total',
                            'spread_home_price','spread_away_price','total_over_price','total_under_price'
                        ]
                        keep_cols = ['game_id'] + [c for c in odds_cols if c in p.columns]
                        if 'game_id' in merged.columns and 'game_id' in p.columns and len(keep_cols) > 1:
                            pf = p[keep_cols].drop_duplicates()
                            merged_odds = merged.merge(pf, on='game_id', how='left', suffixes=('', '_pclose'))
                            # Fill only for finalized rows and where base is missing
                            for c in [c for c in odds_cols if c in merged_odds.columns and f"{c}_pclose" in merged_odds.columns]:
                                base = c
                                alt = f"{c}_pclose"
                                try:
                                    m = finals_mask & merged_odds[base].isna() & merged_odds[alt].notna()
                                except Exception:
                                    # If base doesn't yet exist, create and fill for finals
                                    if base not in merged_odds.columns:
                                        merged_odds[base] = pd.NA
                                    m = finals_mask & merged_odds[alt].notna()
                                if m.any():
                                    merged_odds.loc[m, base] = merged_odds.loc[m, alt]
                            # Promote close_* into canonical fields when only close exists (finals)
                            try:
                                m2 = finals_mask
                                if 'spread_home' in merged_odds.columns and 'close_spread_home' in merged_odds.columns:
                                    mm = m2 & merged_odds['spread_home'].isna() & merged_odds['close_spread_home'].notna()
                                    if mm.any():
                                        merged_odds.loc[mm, 'spread_home'] = merged_odds.loc[mm, 'close_spread_home']
                                if 'total' in merged_odds.columns and 'close_total' in merged_odds.columns:
                                    mm2 = m2 & merged_odds['total'].isna() & merged_odds['close_total'].notna()
                                    if mm2.any():
                                        merged_odds.loc[mm2, 'total'] = merged_odds.loc[mm2, 'close_total']
                                # Ensure market_* aliases reflect best available for finals
                                if 'market_spread_home' not in merged_odds.columns:
                                    merged_odds['market_spread_home'] = pd.NA
                                if 'market_total' not in merged_odds.columns:
                                    merged_odds['market_total'] = pd.NA
                                mm3 = m2 & merged_odds['market_spread_home'].isna()
                                if 'spread_home' in merged_odds.columns:
                                    merged_odds.loc[mm3 & merged_odds['spread_home'].notna(), 'market_spread_home'] = merged_odds.loc[mm3, 'spread_home']
                                if 'close_spread_home' in merged_odds.columns:
                                    mm3b = m2 & merged_odds['market_spread_home'].isna() & merged_odds['close_spread_home'].notna()
                                    merged_odds.loc[mm3b, 'market_spread_home'] = merged_odds.loc[mm3b, 'close_spread_home']
                                mtm = m2 & merged_odds['market_total'].isna()
                                if 'total' in merged_odds.columns:
                                    merged_odds.loc[mtm & merged_odds['total'].notna(), 'market_total'] = merged_odds.loc[mtm, 'total']
                                if 'close_total' in merged_odds.columns:
                                    mtm2 = m2 & merged_odds['market_total'].isna() & merged_odds['close_total'].notna()
                                    merged_odds.loc[mtm2, 'market_total'] = merged_odds.loc[mtm2, 'close_total']
                            except Exception:
                                pass
                            # Drop helper suffixed cols
                            drops = [c for c in merged_odds.columns if c.endswith('_pclose')]
                            if drops:
                                merged_odds = merged_odds.drop(columns=drops)
                            merged = merged_odds
                except Exception:
                    pass
    except Exception:
        pass

    # Final safety: ensure one row per game in the weekly view to avoid duplicate cards.
    try:
        if 'game_id' in merged.columns and not merged['game_id'].isna().all():
            merged = merged.sort_values(['season','week','game_date','date'], axis=0, kind='mergesort') if {
                'season','week'
            }.issubset(merged.columns) else merged
            merged = merged.drop_duplicates(subset=['game_id'], keep='first')
        else:
            # Fallback key when game_id missing: (season, week, home_team, away_team, game_date/date)
            keys = [c for c in ['season','week','home_team','away_team','game_date'] if c in merged.columns]
            if 'game_date' not in keys and 'date' in merged.columns:
                keys.append('date')
            if keys:
                merged = merged.sort_values(keys, axis=0, kind='mergesort')
                merged = merged.drop_duplicates(subset=keys, keep='first')
    except Exception:
        # Non-fatal: return as-is if dedup hits an unexpected issue
        pass

    return merged


def _derive_predictions_from_market(df: pd.DataFrame) -> pd.DataFrame:
    """Derive minimal synthetic prediction columns from available market lines when the
    primary model prediction columns are completely absent or entirely null.

    Logic:
    - Only activates if all of (pred_home_points, pred_away_points, pred_total, pred_margin, prob_home_win)
      are missing or fully null across the frame.
    - Uses coalesced spread (home) and total columns from any of: close_*, market_*, open_*, base names.
    - Points: home = total/2 - spread/2  (spread is home - away from market perspective where home favorites negative)
             away = total - home.
    - Margin: home_points - away_points.
    - Home win probability: derived from moneyline_home & moneyline_away odds if both present.
    Adds columns and sets 'prediction_source' to 'market_synth' where created. If prediction_source already
    exists it will only fill null entries for rows we synthesize.
    """
    if df is None or df.empty:
        return df
    need = ["pred_home_points","pred_away_points","pred_total","pred_margin","prob_home_win"]
    have_any = False
    for c in need:
        if c in df.columns and df[c].notna().any():
            have_any = True
            break
    if have_any:
        return df  # Respect existing model outputs

    work = df.copy()

    def _coalesce(cols: list[str]):
        existing = [c for c in cols if c in work.columns]
        if not existing:
            return pd.Series([None]*len(work), index=work.index)
        out = None
        for c in existing:
            ser = pd.to_numeric(work[c], errors='coerce')
            if out is None:
                out = ser
            else:
                out = out.where(out.notna(), ser)
        return out if out is not None else pd.Series([None]*len(work), index=work.index)

    spread = _coalesce(["close_spread_home","market_spread_home","open_spread_home","spread_home"])
    total = _coalesce(["close_total","market_total","open_total","total"])

    # Derive implied probabilities from moneylines
    mh = work.get("moneyline_home")
    ma = work.get("moneyline_away")
    prob_home = pd.Series([None]*len(work), index=work.index)
    if mh is not None and ma is not None:
        def _american_to_decimal(o):
            try:
                if o is None or (isinstance(o,float) and pd.isna(o)): return None
                o = float(o)
                return (1.0 + 100.0/abs(o)) if o < 0 else (1.0 + o/100.0)
            except Exception:
                return None
        dh = pd.to_numeric(mh, errors='coerce')
        da = pd.to_numeric(ma, errors='coerce')
        for i in work.index:
            h = dh.get(i)
            a = da.get(i)
            if pd.notna(h) and pd.notna(a):
                dec_h = _american_to_decimal(h)
                dec_a = _american_to_decimal(a)
                if dec_h and dec_a:
                    try:
                        ph = (1.0/dec_h) / (1.0/dec_h + 1.0/dec_a)
                        prob_home.at[i] = ph
                    except Exception:
                        pass

    # Points derivation (only where total exists)
    total_exists = total.notna()
    spread_eff = spread.where(spread.notna(), 0.0)
    try:
        home_pts = (total/2.0) - (spread_eff/2.0)
        away_pts = total - home_pts
        margin = home_pts - away_pts
    except Exception:
        home_pts = pd.Series([None]*len(work), index=work.index)
        away_pts = pd.Series([None]*len(work), index=work.index)
        margin = pd.Series([None]*len(work), index=work.index)

    # Apply only where we have totals
    home_pts = home_pts.where(total_exists)
    away_pts = away_pts.where(total_exists)
    margin = margin.where(total_exists)

    if 'pred_home_points' not in work.columns:
        work['pred_home_points'] = home_pts
    if 'pred_away_points' not in work.columns:
        work['pred_away_points'] = away_pts
    if 'pred_total' not in work.columns:
        work['pred_total'] = total
    if 'pred_margin' not in work.columns:
        work['pred_margin'] = margin
    if 'prob_home_win' not in work.columns:
        work['prob_home_win'] = prob_home

    # Tag source
    if 'prediction_source' not in work.columns:
        work['prediction_source'] = None
    # Only set where we actually produced a total (proxy for success)
    synth_mask = total_exists & work['pred_total'].notna()
    work.loc[synth_mask, 'prediction_source'] = work.loc[synth_mask, 'prediction_source'].fillna('market_synth')
    work['derived_from_market'] = synth_mask
    return work


def _attach_model_predictions(view_df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort: fill missing prediction columns for the given rows using trained models.
    - Skips when running on Render (RENDER env true).
    - Uses game_id to align where possible; falls back to (home_team, away_team, game_date/date).
    """
    try:
        if view_df is None or view_df.empty:
            return view_df
        # Always perform odds/weather enrichment FIRST so even disabled prediction inference still yields market data.
        out_base = view_df.copy()
        try:
            from nfl_compare.src.team_normalizer import normalize_team_name as _norm_team
            for _col in ("home_team", "away_team"):
                if _col in out_base.columns:
                    def _safe_norm(v):
                        try:
                            if v is None or (isinstance(v, float) and pd.isna(v)):
                                return v
                            return _norm_team(str(v))
                        except Exception:
                            return v
                    out_base[_col] = out_base[_col].apply(_safe_norm)
        except Exception:
            pass
        # EARLY odds alias setup (may be re-filled later, but ensure columns exist and closers promoted for display)
        try:
            import pandas as _pd
            if 'close_spread_home' in out_base.columns and 'spread_home' in out_base.columns:
                mask = out_base['spread_home'].isna() & out_base['close_spread_home'].notna()
                if mask.any():
                    out_base.loc[mask, 'spread_home'] = out_base.loc[mask, 'close_spread_home']
            if 'close_total' in out_base.columns and 'total' in out_base.columns:
                mask = out_base['total'].isna() & out_base['close_total'].notna()
                if mask.any():
                    out_base.loc[mask, 'total'] = out_base.loc[mask, 'close_total']
            if 'market_spread_home' not in out_base.columns:
                out_base['market_spread_home'] = _pd.NA
            if 'market_total' not in out_base.columns:
                out_base['market_total'] = _pd.NA
            # Fill market_* from base then close
            msk = out_base['market_spread_home'].isna() if 'market_spread_home' in out_base.columns else None
            if msk is not None and 'spread_home' in out_base.columns:
                out_base.loc[msk & out_base['spread_home'].notna(), 'market_spread_home'] = out_base.loc[msk, 'spread_home']
            if 'market_spread_home' in out_base.columns:
                msk2 = out_base['market_spread_home'].isna()
                if 'close_spread_home' in out_base.columns:
                    out_base.loc[msk2 & out_base['close_spread_home'].notna(), 'market_spread_home'] = out_base.loc[msk2, 'close_spread_home']
            msk_t = out_base['market_total'].isna() if 'market_total' in out_base.columns else None
            if msk_t is not None and 'total' in out_base.columns:
                out_base.loc[msk_t & out_base['total'].notna(), 'market_total'] = out_base.loc[msk_t, 'total']
            if 'market_total' in out_base.columns:
                msk_t2 = out_base['market_total'].isna()
                if 'close_total' in out_base.columns:
                    out_base.loc[msk_t2 & out_base['close_total'].notna(), 'market_total'] = out_base.loc[msk_t2, 'close_total']
        except Exception:
            pass
        # After enrichment, record disable flag but don't return yet so normalization runs
        disable_flag = False
        try:
            disable = os.environ.get('DISABLE_ON_REQUEST_PREDICTIONS')
            if disable is None:
                disable = os.environ.get('RENDER', '0')
            if str(disable).strip().lower() in {'1','true','yes','y'}:
                disable_flag = True
        except Exception:
            disable_flag = False
        try:
            from nfl_compare.src.data_sources import load_games as ds_load_games, load_lines
            from nfl_compare.src.weather import load_weather_for_games
            line_cols = ['spread_home','total','moneyline_home','moneyline_away',
                         'spread_home_price','spread_away_price','total_over_price','total_under_price',
                         'close_spread_home','close_total']
            # Merge lines by game_id (preferred), fallback to team-based
            try:
                lines = load_lines()
            except Exception:
                lines = None
            if lines is not None and not getattr(lines, 'empty', True):
                cols_present = [c for c in line_cols if c in lines.columns]
                # Deduplicate right-hand lines by game_id, preferring rows with most filled values
                if 'game_id' in lines.columns:
                    # Prefer newest odds by date when available; tie-break by non-null count
                    keep_cols = ['game_id'] + cols_present + ([c for c in ['date','game_date'] if c in lines.columns])
                    _r_gid = lines[keep_cols].copy()
                    try:
                        import pandas as _pd
                        _r_gid['_nn'] = _r_gid[cols_present].notna().sum(axis=1)
                        if 'date' in _r_gid.columns or 'game_date' in _r_gid.columns:
                            dcol = 'date' if 'date' in _r_gid.columns else 'game_date'
                            _r_gid['_dt'] = _pd.to_datetime(_r_gid[dcol], errors='coerce')
                            _r_gid = _r_gid.sort_values(['_dt','_nn'], ascending=[False, False])
                            _r_gid = _r_gid.drop(columns=['_dt'])
                        else:
                            _r_gid = _r_gid.sort_values(['_nn'], ascending=False)
                        _r_gid = _r_gid.drop_duplicates(['game_id'], keep='first').drop(columns=['_nn'])
                    except Exception:
                        _r_gid = _r_gid.drop_duplicates(['game_id'], keep='first')
                else:
                    _r_gid = None
                if 'game_id' in out_base.columns and _r_gid is not None:
                    out_base = out_base.merge(_r_gid, on='game_id', how='left', suffixes=('', '_ln'))
                # Supplement by season/week + (home_team, away_team) when possible; else by teams only.
                if {'home_team','away_team'}.issubset(set(lines.columns)):
                    if {'season','week'}.issubset(set(lines.columns)) and {'season','week'}.issubset(set(out_base.columns)):
                        keep_cols = ['season','week','home_team','away_team'] + cols_present + ([c for c in ['date','game_date'] if c in lines.columns])
                        sup = lines[keep_cols].copy()
                        try:
                            import pandas as _pd
                            sup['_nn'] = sup[cols_present].notna().sum(axis=1)
                            if 'date' in sup.columns or 'game_date' in sup.columns:
                                dcol = 'date' if 'date' in sup.columns else 'game_date'
                                sup['_dt'] = _pd.to_datetime(sup[dcol], errors='coerce')
                                sup = sup.sort_values(['_dt','_nn'], ascending=[False, False]).drop(columns=['_dt'])
                            else:
                                sup = sup.sort_values(['_nn'], ascending=False)
                            sup = sup.drop_duplicates(['season','week','home_team','away_team'], keep='first').drop(columns=['_nn'])
                        except Exception:
                            sup = sup.drop_duplicates(['season','week','home_team','away_team'], keep='first')
                        out_base = out_base.merge(sup, on=['season','week','home_team','away_team'], how='left', suffixes=('', '_ln2'))
                    else:
                        keep_cols = ['home_team','away_team'] + cols_present + ([c for c in ['date','game_date'] if c in lines.columns])
                        sup = lines[keep_cols].copy()
                        try:
                            import pandas as _pd
                            sup['_nn'] = sup[cols_present].notna().sum(axis=1)
                            if 'date' in sup.columns or 'game_date' in sup.columns:
                                dcol = 'date' if 'date' in sup.columns else 'game_date'
                                sup['_dt'] = _pd.to_datetime(sup[dcol], errors='coerce')
                                sup = sup.sort_values(['_dt','_nn'], ascending=[False, False]).drop(columns=['_dt'])
                            else:
                                sup = sup.sort_values(['_nn'], ascending=False)
                            sup = sup.drop_duplicates(['home_team','away_team'], keep='first').drop(columns=['_nn'])
                        except Exception:
                            sup = sup.drop_duplicates(['home_team','away_team'], keep='first')
                        out_base = out_base.merge(sup, on=['home_team','away_team'], how='left', suffixes=('', '_ln2'))
                # Fill from any suffix variants (and create columns if missing)
                for c in line_cols:
                    c1, c2 = f"{c}_ln", f"{c}_ln2"
                    has_base = c in out_base.columns
                    v1 = out_base[c1] if c1 in out_base.columns else None
                    v2 = out_base[c2] if c2 in out_base.columns else None
                    if has_base:
                        if v1 is not None:
                            out_base[c] = out_base[c].where(out_base[c].notna(), v1)
                        if v2 is not None:
                            out_base[c] = out_base[c].where(out_base[c].notna(), v2)
                    else:
                        # Create the base column from right-hand values when not present
                        if v1 is not None:
                            out_base[c] = v1
                            has_base = True
                        if (not has_base) and v2 is not None:
                            out_base[c] = v2
                # Drop helper suffix columns
                drop_cols = [c for c in out_base.columns if c.endswith('_ln') or c.endswith('_ln2')]
                if drop_cols:
                    out_base = out_base.drop(columns=drop_cols)
            # Always attempt a raw CSV fallback merge to fill any remaining gaps (per-row), even if load_lines() failed
            import pandas as _pd
            # Use the same DATA_DIR as the rest of the app (respects NFL_DATA_DIR)
            csv_fp = DATA_DIR / 'lines.csv'
            df_csv_fb = None
            if csv_fp.exists():
                try:
                    df_csv_fb = _pd.read_csv(csv_fp)
                except Exception:
                    df_csv_fb = None
            else:
                # Try top-level ./data/lines.csv as a fallback (when running locally with data/ checked out at root)
                alt_lines_fp = BASE_DIR / 'data' / 'lines.csv'
                if alt_lines_fp.exists():
                    try:
                        df_csv_fb = _pd.read_csv(alt_lines_fp)
                    except Exception:
                        df_csv_fb = None
            if df_csv_fb is not None:
                try:
                    # Normalize team names in fallback to match app conventions
                    try:
                        from nfl_compare.src.team_normalizer import normalize_team_name as _norm_team
                        if 'home_team' in df_csv_fb.columns:
                            df_csv_fb['home_team'] = df_csv_fb['home_team'].astype(str).apply(_norm_team)
                        if 'away_team' in df_csv_fb.columns:
                            df_csv_fb['away_team'] = df_csv_fb['away_team'].astype(str).apply(_norm_team)
                    except Exception:
                        pass
                    # Align dtypes for season/week if present
                    for _col in ('season','week'):
                        if _col in df_csv_fb.columns:
                            df_csv_fb[_col] = _pd.to_numeric(df_csv_fb[_col], errors='coerce').astype('Int64')
                        if _col in out_base.columns:
                            out_base[_col] = _pd.to_numeric(out_base[_col], errors='coerce').astype('Int64')
                    cols_present_fb = [c for c in line_cols if c in df_csv_fb.columns]
                    # Cast game_id to str on both sides when present
                    if 'game_id' in out_base.columns:
                        out_base['game_id'] = out_base['game_id'].astype(str)
                    if 'game_id' in df_csv_fb.columns:
                        df_csv_fb['game_id'] = df_csv_fb['game_id'].astype(str)
                    # Prepare de-duplicated right-hand slices with priority by non-null counts
                    if 'game_id' in df_csv_fb.columns:
                        keep_cols = ['game_id'] + cols_present_fb + ([c for c in ['date','game_date'] if c in df_csv_fb.columns])
                        _fb_gid = df_csv_fb[keep_cols].copy()
                        try:
                            import pandas as _pd
                            _fb_gid['_nn'] = _fb_gid[cols_present_fb].notna().sum(axis=1)
                            if 'date' in _fb_gid.columns or 'game_date' in _fb_gid.columns:
                                dcol = 'date' if 'date' in _fb_gid.columns else 'game_date'
                                _fb_gid['_dt'] = _pd.to_datetime(_fb_gid[dcol], errors='coerce')
                                _fb_gid = _fb_gid.sort_values(['_dt','_nn'], ascending=[False, False]).drop(columns=['_dt'])
                            else:
                                _fb_gid = _fb_gid.sort_values(['_nn'], ascending=False)
                            _fb_gid = _fb_gid.drop_duplicates(['game_id'], keep='first').drop(columns=['_nn'])
                        except Exception:
                            _fb_gid = _fb_gid.drop_duplicates(['game_id'], keep='first')
                    else:
                        _fb_gid = None
                    if 'game_id' in out_base.columns and _fb_gid is not None:
                        out_base = out_base.merge(
                            _fb_gid,
                            on='game_id',
                            how='left',
                            suffixes=('', '_lnfb')
                        )
                    # Also try season/week/home/away even if game_id merge ran
                    if {'season','week','home_team','away_team'}.issubset(set(df_csv_fb.columns)) and {'season','week','home_team','away_team'}.issubset(set(out_base.columns)):
                        keep_cols = ['season','week','home_team','away_team'] + cols_present_fb + ([c for c in ['date','game_date'] if c in df_csv_fb.columns])
                        _fb_sw = df_csv_fb[keep_cols].copy()
                        try:
                            import pandas as _pd
                            _fb_sw['_nn'] = _fb_sw[cols_present_fb].notna().sum(axis=1)
                            if 'date' in _fb_sw.columns or 'game_date' in _fb_sw.columns:
                                dcol = 'date' if 'date' in _fb_sw.columns else 'game_date'
                                _fb_sw['_dt'] = _pd.to_datetime(_fb_sw[dcol], errors='coerce')
                                _fb_sw = _fb_sw.sort_values(['_dt','_nn'], ascending=[False, False]).drop(columns=['_dt'])
                            else:
                                _fb_sw = _fb_sw.sort_values(['_nn'], ascending=False)
                            _fb_sw = _fb_sw.drop_duplicates(['season','week','home_team','away_team'], keep='first').drop(columns=['_nn'])
                        except Exception:
                            _fb_sw = _fb_sw.drop_duplicates(['season','week','home_team','away_team'], keep='first')
                        out_base = out_base.merge(
                            _fb_sw,
                            on=['season','week','home_team','away_team'],
                            how='left',
                            suffixes=('', '_lnfb2')
                        )
                    # Finally try by teams only
                    if {'home_team','away_team'}.issubset(df_csv_fb.columns):
                        keep_cols = ['home_team','away_team'] + cols_present_fb + ([c for c in ['date','game_date'] if c in df_csv_fb.columns])
                        _fb_h2h = df_csv_fb[keep_cols].copy()
                        try:
                            import pandas as _pd
                            _fb_h2h['_nn'] = _fb_h2h[cols_present_fb].notna().sum(axis=1)
                            if 'date' in _fb_h2h.columns or 'game_date' in _fb_h2h.columns:
                                dcol = 'date' if 'date' in _fb_h2h.columns else 'game_date'
                                _fb_h2h['_dt'] = _pd.to_datetime(_fb_h2h[dcol], errors='coerce')
                                _fb_h2h = _fb_h2h.sort_values(['_dt','_nn'], ascending=[False, False]).drop(columns=['_dt'])
                            else:
                                _fb_h2h = _fb_h2h.sort_values(['_nn'], ascending=False)
                            _fb_h2h = _fb_h2h.drop_duplicates(['home_team','away_team'], keep='first').drop(columns=['_nn'])
                        except Exception:
                            _fb_h2h = _fb_h2h.drop_duplicates(['home_team','away_team'], keep='first')
                        out_base = out_base.merge(
                            _fb_h2h,
                            on=['home_team','away_team'],
                            how='left',
                            suffixes=('', '_lnfb3')
                        )
                    # Per-column fill from fallback where base is missing
                    for c in line_cols:
                        # Fill priority: _lnfb (game_id), then _lnfb2 (season/week/home/away), then _lnfb3 (teams)
                        for suf in ('_lnfb', '_lnfb2', '_lnfb3'):
                            cfb = f"{c}{suf}"
                            if c in out_base.columns and cfb in out_base.columns:
                                out_base[c] = out_base[c].where(out_base[c].notna(), out_base[cfb])
                            elif cfb in out_base.columns and c not in out_base.columns:
                                out_base[c] = out_base[cfb]
                    drop_fb = [c for c in out_base.columns if c.endswith('_lnfb') or c.endswith('_lnfb2') or c.endswith('_lnfb3')]
                    if drop_fb:
                        out_base = out_base.drop(columns=drop_fb)
                except Exception:
                    pass
            # After all merges/fallbacks, refresh base and market aliases to reflect latest data
            try:
                import pandas as _pd
                # Backfill base fields from close_* if still missing
                if 'close_spread_home' in out_base.columns:
                    if 'spread_home' not in out_base.columns:
                        out_base['spread_home'] = _pd.NA
                    m_sp = out_base['spread_home'].isna()
                    out_base.loc[m_sp & out_base['close_spread_home'].notna(), 'spread_home'] = out_base.loc[m_sp, 'close_spread_home']
                if 'close_total' in out_base.columns:
                    if 'total' not in out_base.columns:
                        out_base['total'] = _pd.NA
                    m_t = out_base['total'].isna()
                    out_base.loc[m_t & out_base['close_total'].notna(), 'total'] = out_base.loc[m_t, 'close_total']

                # Ensure market_* reflect latest base/close fields (prefer base, then close)
                if 'market_spread_home' not in out_base.columns:
                    out_base['market_spread_home'] = _pd.NA
                if 'market_total' not in out_base.columns:
                    out_base['market_total'] = _pd.NA
                # Overwrite market_* with best available current values
                out_base['market_spread_home'] = out_base.get('spread_home') if 'spread_home' in out_base.columns else out_base['market_spread_home']
                out_base['market_total'] = out_base.get('total') if 'total' in out_base.columns else out_base['market_total']
                # Where still missing, use close_* as secondary
                if 'close_spread_home' in out_base.columns:
                    m2 = out_base['market_spread_home'].isna()
                    out_base.loc[m2 & out_base['close_spread_home'].notna(), 'market_spread_home'] = out_base.loc[m2, 'close_spread_home']
                if 'close_total' in out_base.columns:
                    m2t = out_base['market_total'].isna()
                    out_base.loc[m2t & out_base['close_total'].notna(), 'market_total'] = out_base.loc[m2t, 'close_total']
            except Exception:
                pass
            # Last-resort: for any rows still missing odds/close fields, directly lookup in fallback CSV by keys and assign
            try:
                if 'df_csv_fb' in locals() and isinstance(df_csv_fb, _pd.DataFrame) and not df_csv_fb.empty:
                    key_cols_all = ['season','week','home_team','away_team']
                    # Build quick indices for performance (small data anyway)
                    df_by_gid = None
                    if 'game_id' in df_csv_fb.columns:
                        df_by_gid = df_csv_fb.set_index('game_id')
                    # Identify rows with any of the main odds missing
                    odds_main = ['moneyline_home','moneyline_away','close_spread_home','close_total']
                    need_fill = out_base.index[out_base[odds_main].isna().any(axis=1)].tolist() if all(c in out_base.columns for c in odds_main) else []
                    for ridx in need_fill:
                        row = out_base.loc[ridx]
                        cand = None
                        swapped = False
                        # Try by game_id
                        gid = str(row.get('game_id')) if 'game_id' in out_base.columns else None
                        if gid and df_by_gid is not None and gid in df_by_gid.index:
                            cand = df_by_gid.loc[gid]
                            # If duplicate game_id rows exist, take the first
                            if isinstance(cand, _pd.DataFrame):
                                cand = cand.iloc[0]
                        # Try by season/week/home/away
                        if cand is None and all(c in out_base.columns for c in key_cols_all) and all(c in df_csv_fb.columns for c in key_cols_all):
                            mask = (
                                (df_csv_fb['season'] == row.get('season')) &
                                (df_csv_fb['week'] == row.get('week')) &
                                (df_csv_fb['home_team'] == row.get('home_team')) &
                                (df_csv_fb['away_team'] == row.get('away_team'))
                            )
                            sub = df_csv_fb[mask]
                            if len(sub) >= 1:
                                # pick the row with most non-null among priority fields
                                priority_cols = ['moneyline_home','moneyline_away','close_spread_home','close_total','spread_home','total']
                                sub = sub.copy()
                                sub['_nn'] = sub[ [c for c in priority_cols if c in sub.columns] ].notna().sum(axis=1)
                                cand = sub.sort_values(['_nn'], ascending=False).iloc[0]
                            # If not found or ambiguous, try swapped home/away (defensive)
                            if cand is None:
                                mask_sw = (
                                    (df_csv_fb['season'] == row.get('season')) &
                                    (df_csv_fb['week'] == row.get('week')) &
                                    (df_csv_fb['home_team'] == row.get('away_team')) &
                                    (df_csv_fb['away_team'] == row.get('home_team'))
                                )
                                sub_sw = df_csv_fb[mask_sw]
                                if len(sub_sw) >= 1:
                                    priority_cols = ['moneyline_home','moneyline_away','close_spread_home','close_total','spread_home','total']
                                    sub_sw = sub_sw.copy()
                                    sub_sw['_nn'] = sub_sw[ [c for c in priority_cols if c in sub_sw.columns] ].notna().sum(axis=1)
                                    cand = sub_sw.sort_values(['_nn'], ascending=False).iloc[0]
                                    swapped = True
                        # Try by teams only
                        if cand is None and {'home_team','away_team'}.issubset(df_csv_fb.columns):
                            mask = (
                                (df_csv_fb['home_team'] == row.get('home_team')) &
                                (df_csv_fb['away_team'] == row.get('away_team'))
                            )
                            sub = df_csv_fb[mask]
                            if len(sub) >= 1:
                                priority_cols = ['moneyline_home','moneyline_away','close_spread_home','close_total','spread_home','total']
                                sub = sub.copy()
                                sub['_nn'] = sub[ [c for c in priority_cols if c in sub.columns] ].notna().sum(axis=1)
                                cand = sub.sort_values(['_nn'], ascending=False).iloc[0]
                            if cand is None:
                                mask_sw = (
                                    (df_csv_fb['home_team'] == row.get('away_team')) &
                                    (df_csv_fb['away_team'] == row.get('home_team'))
                                )
                                sub_sw = df_csv_fb[mask_sw]
                                if len(sub_sw) >= 1:
                                    priority_cols = ['moneyline_home','moneyline_away','close_spread_home','close_total','spread_home','total']
                                    sub_sw = sub_sw.copy()
                                    sub_sw['_nn'] = sub_sw[ [c for c in priority_cols if c in sub_sw.columns] ].notna().sum(axis=1)
                                    cand = sub_sw.sort_values(['_nn'], ascending=False).iloc[0]
                                    swapped = True
                        if cand is not None:
                            # If still a DataFrame (edge case), reduce to first row
                            if isinstance(cand, _pd.DataFrame) and not cand.empty:
                                cand = cand.iloc[0]
                            # When matched on swapped orientation, swap/signal-correct fields accordingly
                            if swapped:
                                # Moneylines swap
                                if 'moneyline_home' in out_base.columns:
                                    out_base.at[ridx, 'moneyline_home'] = cand.get('moneyline_away') if _pd.isna(out_base.at[ridx, 'moneyline_home']) else out_base.at[ridx, 'moneyline_home']
                                if 'moneyline_away' in out_base.columns:
                                    out_base.at[ridx, 'moneyline_away'] = cand.get('moneyline_home') if _pd.isna(out_base.at[ridx, 'moneyline_away']) else out_base.at[ridx, 'moneyline_away']
                                # Spread sign flip for home spread
                                if 'spread_home' in out_base.columns and 'spread_home' in cand.index and _pd.isna(out_base.at[ridx, 'spread_home']):
                                    try:
                                        out_base.at[ridx, 'spread_home'] = -float(cand['spread_home']) if cand['spread_home'] is not None and not _pd.isna(cand['spread_home']) else out_base.at[ridx, 'spread_home']
                                    except Exception:
                                        pass
                                if 'close_spread_home' in out_base.columns and 'close_spread_home' in cand.index and _pd.isna(out_base.at[ridx, 'close_spread_home']):
                                    try:
                                        out_base.at[ridx, 'close_spread_home'] = -float(cand['close_spread_home']) if 'close_spread_home' in cand.index and cand['close_spread_home'] is not None and not _pd.isna(cand['close_spread_home']) else out_base.at[ridx, 'close_spread_home']
                                    except Exception:
                                        pass
                                # Prices swap
                                if 'spread_home_price' in out_base.columns and 'spread_away_price' in cand.index and _pd.isna(out_base.at[ridx, 'spread_home_price']):
                                    out_base.at[ridx, 'spread_home_price'] = cand.get('spread_away_price')
                                if 'spread_away_price' in out_base.columns and 'spread_home_price' in cand.index and _pd.isna(out_base.at[ridx, 'spread_away_price']):
                                    out_base.at[ridx, 'spread_away_price'] = cand.get('spread_home_price')
                                # Totals are symmetric
                                if 'total' in out_base.columns and 'total' in cand.index and _pd.isna(out_base.at[ridx, 'total']):
                                    out_base.at[ridx, 'total'] = cand.get('total')
                                if 'close_total' in out_base.columns and 'close_total' in cand.index and _pd.isna(out_base.at[ridx, 'close_total']):
                                    out_base.at[ridx, 'close_total'] = cand.get('close_total')
                                if 'total_over_price' in out_base.columns and 'total_over_price' in cand.index and _pd.isna(out_base.at[ridx, 'total_over_price']):
                                    out_base.at[ridx, 'total_over_price'] = cand.get('total_over_price')
                                if 'total_under_price' in out_base.columns and 'total_under_price' in cand.index and _pd.isna(out_base.at[ridx, 'total_under_price']):
                                    out_base.at[ridx, 'total_under_price'] = cand.get('total_under_price')
                            else:
                                for c in line_cols:
                                    if c in out_base.columns and _pd.isna(out_base.at[ridx, c]) and c in cand.index:
                                        out_base.at[ridx, c] = cand[c]
                                # Also set market_* aliases if base fields missing
                                if 'spread_home' in cand.index:
                                    if 'market_spread_home' in out_base.columns and _pd.isna(out_base.at[ridx, 'market_spread_home']):
                                        out_base.at[ridx, 'market_spread_home'] = cand.get('spread_home')
                                if 'total' in cand.index:
                                    if 'market_total' in out_base.columns and _pd.isna(out_base.at[ridx, 'market_total']):
                                        out_base.at[ridx, 'market_total'] = cand.get('total')
            except Exception:
                pass
            # Final de-duplication: ensure at most one row per game after all enrichments
            try:
                line_cols_final = ['moneyline_home','moneyline_away','spread_home','total','spread_home_price','spread_away_price','total_over_price','total_under_price','close_spread_home','close_total','market_spread_home','market_total']
                keep_keys = [k for k in ['season','week','home_team','away_team'] if k in out_base.columns]
                if 'game_id' in out_base.columns and out_base['game_id'].notna().any():
                    tmp = out_base.copy()
                    try:
                        import pandas as _pd
                        tmp['_nn'] = tmp[[c for c in line_cols_final if c in tmp.columns]].notna().sum(axis=1)
                        # Prefer latest date when available, then non-null count
                        if 'date' in tmp.columns or 'game_date' in tmp.columns:
                            dcol = 'date' if 'date' in tmp.columns else 'game_date'
                            tmp['_dt'] = _pd.to_datetime(tmp[dcol], errors='coerce')
                            tmp = tmp.sort_values(['_dt','_nn'], ascending=[False, False]).drop(columns=['_dt'])
                        else:
                            tmp = tmp.sort_values(['_nn'], ascending=False)
                        tmp = tmp.drop_duplicates(['game_id'], keep='first').drop(columns=['_nn'])
                    except Exception:
                        tmp = tmp.drop_duplicates(['game_id'], keep='first')
                    out_base = tmp
                elif len(keep_keys) == 4:
                    tmp = out_base.copy()
                    try:
                        import pandas as _pd
                        tmp['_nn'] = tmp[[c for c in line_cols_final if c in tmp.columns]].notna().sum(axis=1)
                        if 'date' in tmp.columns or 'game_date' in tmp.columns:
                            dcol = 'date' if 'date' in tmp.columns else 'game_date'
                            tmp['_dt'] = _pd.to_datetime(tmp[dcol], errors='coerce')
                            tmp = tmp.sort_values(['_dt','_nn'], ascending=[False, False]).drop(columns=['_dt'])
                        else:
                            tmp = tmp.sort_values(['_nn'], ascending=False)
                        tmp = tmp.drop_duplicates(keep_keys, keep='first').drop(columns=['_nn'])
                    except Exception:
                        tmp = tmp.drop_duplicates(keep_keys, keep='first')
                    out_base = tmp
            except Exception:
                pass
            # Weather/stadium (optional)
            try:
                games_all = ds_load_games()
                wx = load_weather_for_games(games_all)
            except Exception:
                wx = None
            if wx is not None and not getattr(wx, 'empty', True):
                wx_cols = ['game_id','date','home_team','away_team','wx_temp_f','wx_wind_mph','wx_precip_pct','roof','surface','neutral_site']
                keep = [c for c in wx_cols if c in wx.columns]
                if keep:
                    out_base = out_base.merge(wx[keep], on=[c for c in ['game_id','date','home_team','away_team'] if c in out_base.columns and c in wx.columns], how='left', suffixes=('', '_wx'))
                    # Prefer non-null base, then fill from wx
                    for c in ['wx_temp_f','wx_wind_mph','wx_precip_pct','roof','surface','neutral_site']:
                        cwx = f"{c}_wx"
                        if c in out_base.columns and cwx in out_base.columns:
                            out_base[c] = out_base[c].where(out_base[c].notna(), out_base[cwx])
                    drop_wx = [c for c in out_base.columns if c.endswith('_wx')]
                    if drop_wx:
                        out_base = out_base.drop(columns=drop_wx)
        except Exception:
            pass

        # If running on Render/minimal, skip heavy model predictions but keep enriched odds/weather
        if str(os.environ.get("RENDER", "").lower()) in {"1","true","yes"}:
            # Fallback: if finals have close lines but missing moneylines, derive approximate MLs using spread heuristic
            try:
                import numpy as np
                if {'home_score','away_score','moneyline_home','moneyline_away','close_spread_home'}.issubset(out_base.columns):
                    def _approx_ml_from_spread(sp):
                        # Simple mapping: convert spread to prob via logistic, then to American odds
                        try:
                            sp = float(sp)
                        except Exception:
                            return (None, None)
                        # scale tuned loosely for NFL; avoids extreme numbers
                        sigma = float(os.environ.get('NFL_SPREAD_PROB_SIGMA', '7.0'))
                        import math
                        p_home = 1.0/(1.0+math.exp(-( -sp)/sigma))  # negative spread favors home
                        p_home = min(max(p_home, 0.05), 0.95)
                        # Convert prob to fair American odds
                        def _prob_to_american(p):
                            if p <= 0 or p >= 1:
                                return None
                            if p >= 0.5:
                                return int(round(-p/(1-p)*100))
                            else:
                                return int(round((1-p)/p*100))
                        return (_prob_to_american(1-p_home), _prob_to_american(p_home))
                    finals = out_base[(out_base['home_score'].notna()) & (out_base['away_score'].notna())]
                    for idx, rr in finals.iterrows():
                        if (pd.isna(rr.get('moneyline_home')) or pd.isna(rr.get('moneyline_away'))):
                            sp = rr.get('close_spread_home') if pd.notna(rr.get('close_spread_home')) else rr.get('spread_home')
                            if pd.notna(sp):
                                ml_away, ml_home = _approx_ml_from_spread(sp)
                                if pd.isna(rr.get('moneyline_home')) and ml_home is not None:
                                    out_base.at[idx, 'moneyline_home'] = ml_home
                                if pd.isna(rr.get('moneyline_away')) and ml_away is not None:
                                    out_base.at[idx, 'moneyline_away'] = ml_away
            except Exception:
                pass
            # FINAL SAFETY FILL: direct game_id mapping from lines.csv (after all other fallbacks)
            try:
                import pandas as _pd
                csv_fp = BASE_DIR / 'nfl_compare' / 'data' / 'lines.csv'
                if csv_fp.exists() and 'game_id' in out_base.columns:
                    df_final = _pd.read_csv(csv_fp)
                    # Normalize team names to be consistent (not strictly needed for game_id)
                    try:
                        from nfl_compare.src.team_normalizer import normalize_team_name as _norm_team
                        if 'home_team' in df_final.columns:
                            df_final['home_team'] = df_final['home_team'].astype(str).apply(_norm_team)
                        if 'away_team' in df_final.columns:
                            df_final['away_team'] = df_final['away_team'].astype(str).apply(_norm_team)
                    except Exception:
                        pass
                    if 'game_id' in df_final.columns:
                        df_final['game_id'] = df_final['game_id'].astype(str)
                    line_cols_final = ['moneyline_home','moneyline_away','spread_home','total','spread_home_price','spread_away_price','total_over_price','total_under_price','close_spread_home','close_total']
                    present_line_cols = [c for c in line_cols_final if c in df_final.columns]
                    # Build quick index by game_id
                    df_g = df_final.set_index('game_id') if 'game_id' in df_final.columns else None
                    if df_g is not None:
                        # Rows needing any fill
                        need_idx = out_base.index[[any(pd.isna(out_base.at[i, c]) for c in line_cols_final if c in out_base.columns) for i in out_base.index]]
                        for i in need_idx:
                            gid = str(out_base.at[i, 'game_id']) if pd.notna(out_base.at[i, 'game_id']) else None
                            if gid and gid in df_g.index:
                                cand = df_g.loc[gid]
                                if isinstance(cand, _pd.DataFrame):
                                    cand = cand.iloc[0]
                                for c in present_line_cols:
                                    if c in out_base.columns and pd.isna(out_base.at[i, c]) and c in cand.index:
                                        try:
                                            out_base.at[i, c] = cand[c]
                                        except Exception:
                                            pass
                                # Also set market_* aliases if absent
                                if 'spread_home' in cand.index and 'market_spread_home' in out_base.columns and pd.isna(out_base.at[i, 'market_spread_home']):
                                    out_base.at[i, 'market_spread_home'] = cand.get('spread_home')
                                if 'total' in cand.index and 'market_total' in out_base.columns and pd.isna(out_base.at[i, 'market_total']):
                                    out_base.at[i, 'market_total'] = cand.get('total')
            except Exception:
                pass
            # Post-merge normalization: promote close_* lines into canonical/market fields when only close values exist
            try:
                import pandas as _pd
                # Ensure market alias columns exist for downstream display logic
                if 'market_spread_home' not in out_base.columns:
                    try:
                        out_base['market_spread_home'] = _pd.NA
                    except Exception:
                        pass
                if 'market_total' not in out_base.columns:
                    try:
                        out_base['market_total'] = _pd.NA
                    except Exception:
                        pass
                # Promote close_spread_home -> spread_home if base missing
                if {'spread_home','close_spread_home'}.issubset(out_base.columns):
                    mask = out_base['spread_home'].isna() & out_base['close_spread_home'].notna()
                    if mask.any():
                        out_base.loc[mask, 'spread_home'] = out_base.loc[mask, 'close_spread_home']
                # Promote close_total -> total if base missing
                if {'total','close_total'}.issubset(out_base.columns):
                    mask = out_base['total'].isna() & out_base['close_total'].notna()
                    if mask.any():
                        out_base.loc[mask, 'total'] = out_base.loc[mask, 'close_total']
                # Ensure market_* aliases populated (prefer existing market -> base -> close)
                if 'market_spread_home' in out_base.columns:
                    m_mask = out_base['market_spread_home'].isna()
                    if m_mask.any():
                        # fill from spread_home first
                        fill1 = out_base.loc[m_mask, 'spread_home']
                        out_base.loc[m_mask & fill1.notna(), 'market_spread_home'] = fill1[fill1.notna()]
                        # then from close_spread_home
                        m_mask2 = out_base['market_spread_home'].isna()
                        if 'close_spread_home' in out_base.columns and m_mask2.any():
                            fill2 = out_base.loc[m_mask2, 'close_spread_home']
                            out_base.loc[m_mask2 & fill2.notna(), 'market_spread_home'] = fill2[fill2.notna()]
                if 'market_total' in out_base.columns:
                    m_mask = out_base['market_total'].isna()
                    if m_mask.any():
                        fill1 = out_base.loc[m_mask, 'total'] if 'total' in out_base.columns else None
                        if fill1 is not None:
                            out_base.loc[m_mask & fill1.notna(), 'market_total'] = fill1[fill1.notna()]
                        m_mask2 = out_base['market_total'].isna()
                        if 'close_total' in out_base.columns and m_mask2.any():
                            fill2 = out_base.loc[m_mask2, 'close_total']
                            out_base.loc[m_mask2 & fill2.notna(), 'market_total'] = fill2[fill2.notna()]
                # One-time log for any games still missing all spread or all total indicators
                try:
                    spread_cols = [c for c in ['spread_home','close_spread_home','market_spread_home','open_spread_home'] if c in out_base.columns]
                    total_cols = [c for c in ['total','close_total','market_total','open_total'] if c in out_base.columns]
                    missing_spread = []
                    missing_total = []
                    if spread_cols:
                        mask = out_base[spread_cols].isna().all(axis=1)
                        if mask.any():
                            missing_spread = out_base.loc[mask, 'game_id'].astype(str).tolist() if 'game_id' in out_base.columns else []
                    if total_cols:
                        mask = out_base[total_cols].isna().all(axis=1)
                        if mask.any():
                            missing_total = out_base.loc[mask, 'game_id'].astype(str).tolist() if 'game_id' in out_base.columns else []
                    if missing_spread:
                        _log_once('missing_market_spread', f"Games missing spread data after normalization: {missing_spread}")
                    if missing_total:
                        _log_once('missing_market_total', f"Games missing total data after normalization: {missing_total}")
                except Exception:
                    pass
            except Exception:
                pass
            # If predictions disabled, return enriched odds-only frame now
            if disable_flag:
                return out_base
            return out_base

        # Lazy imports from package
        from nfl_compare.src.data_sources import load_games as ds_load_games, load_team_stats, load_lines
        from nfl_compare.src.features import merge_features
        from nfl_compare.src.weather import load_weather_for_games
        from nfl_compare.src.models import predict as model_predict

        # Load base data and models
        games = ds_load_games()
        stats = load_team_stats()
        lines = load_lines()
        try:
            models = joblib_load(BASE_DIR / 'nfl_compare' / 'models' / 'nfl_models.joblib')
        except Exception:
            # Models not available; preserve and return the enriched odds/weather frame
            return out_base  # models not available
        try:
            wx = load_weather_for_games(games)
        except Exception:
            wx = None
        feat = merge_features(games, stats, lines, wx)
        if feat is None or feat.empty:
            return out_base
        # Filter features to the rows in view_df (exclude completed/final games to keep historical predictions locked)
        vf = out_base.copy()
        # Determine which rows actually need predictions (any key pred_* missing), but NEVER for finals
        try:
            finals_mask = pd.Series(False, index=vf.index)
            if {'home_score','away_score'}.issubset(vf.columns):
                finals_mask = pd.to_numeric(vf['home_score'], errors='coerce').notna() & pd.to_numeric(vf['away_score'], errors='coerce').notna()
        except Exception:
            finals_mask = pd.Series(False, index=vf.index)
        # Build a missing-any mask across core prediction fields
        missing_any = pd.Series(False, index=vf.index)
        core_pred_fields = ["pred_home_points","pred_away_points","pred_total","pred_home_win_prob"]
        # If alternate naming is used, consider it satisfied
        alt_prob_col = "prob_home_win"
        for col in core_pred_fields:
            if col in vf.columns:
                missing_any = missing_any | vf[col].isna()
            else:
                # If the only missing core is pred_home_win_prob but prob_home_win exists, treat as not missing
                if col == "pred_home_win_prob" and alt_prob_col in vf.columns:
                    # consider satisfied where alt prob is present
                    missing_any = missing_any | vf[alt_prob_col].isna()
                else:
                    # Column entirely absent -> considered missing for non-final rows
                    missing_any = True
        # Need predictions only for non-final rows which are missing any core predictions
        need_mask = (~finals_mask) & missing_any
        vf_pred = vf.loc[need_mask].copy()
        if vf_pred.empty:
            return out_base

        if 'game_id' in vf_pred.columns and 'game_id' in feat.columns:
            keys = vf_pred['game_id'].dropna().astype(str).unique().tolist()
            sub = feat[feat['game_id'].astype(str).isin(keys)].copy()
        else:
            # fallback by match
            key_cols = [c for c in ['season','week','home_team','away_team'] if c in vf_pred.columns and c in feat.columns]
            if key_cols:
                sub = feat.merge(vf_pred[key_cols].drop_duplicates(), on=key_cols, how='inner')
            else:
                sub = feat
        if sub.empty:
            return out_base
        # Run model predictions
        pred = model_predict(models, sub)
        if pred is None or pred.empty:
            return out_base
        # Select columns to merge back
        keep_cols = [c for c in pred.columns if c.startswith('pred_') or c.startswith('prob_')] + ['game_id','home_team','away_team']
        pred_keep = pred[[c for c in keep_cols if c in pred.columns]].copy()
        # Merge back into view_df without overwriting existing non-null prediction values
        if 'game_id' in vf_pred.columns and 'game_id' in pred_keep.columns and pred_keep['game_id'].notna().any():
            merged_partial = vf_pred.merge(pred_keep, on='game_id', how='left', suffixes=('', '_m'))
        else:
            merged_partial = vf_pred.merge(pred_keep, on=[c for c in ['home_team','away_team'] if c in vf_pred.columns and c in pred_keep.columns], how='left', suffixes=('', '_m'))
        # Fill nulls from _m columns (do not overwrite existing non-null values)
        for col in ['pred_home_points','pred_away_points','pred_total','pred_margin','pred_q1_total','pred_q2_total','pred_q3_total','pred_q4_total','pred_1h_total','pred_2h_total','prob_home_win','pred_home_win_prob']:
            base = col
            alt = f"{col}_m"
            if base in merged_partial.columns and alt in merged_partial.columns:
                merged_partial[base] = merged_partial[base].fillna(merged_partial[alt])
        # Also fill odds/lines/weather fields from features frame for completeness
        line_cols = ['spread_home','total','moneyline_home','moneyline_away','spread_home_price','spread_away_price','total_over_price','total_under_price','close_spread_home','close_total',
                     'wx_temp_f','wx_wind_mph','wx_precip_pct','roof','surface','neutral_site']
        feat_keep = feat[['game_id','home_team','away_team'] + [c for c in line_cols if c in feat.columns]].copy()
        if 'game_id' in merged_partial.columns and 'game_id' in feat_keep.columns and feat_keep['game_id'].notna().any():
            merged2_partial = merged_partial.merge(feat_keep, on='game_id', how='left', suffixes=('', '_f'))
        else:
            merged2_partial = merged_partial.merge(feat_keep, on=[c for c in ['home_team','away_team'] if c in merged_partial.columns and c in feat_keep.columns], how='left', suffixes=('', '_f'))
        for col in [c for c in line_cols if c in merged2_partial.columns and f"{c}_f" in merged2_partial.columns]:
            merged2_partial[col] = merged2_partial[col].where(merged2_partial[col].notna(), merged2_partial[f"{col}_f"]) 
        # drop helper cols on partial
        drop_cols2 = [c for c in merged2_partial.columns if c.endswith('_m') or c.endswith('_f')]
        merged2_partial = merged2_partial.drop(columns=drop_cols2)
    # Stitch partial predictions back with untouched finals
        out = vf.copy()
        if 'game_id' in out.columns and 'game_id' in merged2_partial.columns and merged2_partial['game_id'].notna().any():
            out = out.merge(merged2_partial, on=[c for c in out.columns if c in merged2_partial.columns and c in ['game_id']], how='left', suffixes=('', '_new'))
        else:
            join_keys = [c for c in ['season','week','home_team','away_team'] if c in out.columns and c in merged2_partial.columns]
            out = out.merge(merged2_partial, on=join_keys, how='left', suffixes=('', '_new'))
        # For prediction fields, fill missing values from newly computed values (keep existing non-null values intact)
        for col in ['pred_home_points','pred_away_points','pred_total','pred_margin','pred_q1_total','pred_q2_total','pred_q3_total','pred_q4_total','pred_1h_total','pred_2h_total','prob_home_win','pred_home_win_prob']:
            if f"{col}_new" in out.columns:
                out[col] = out[col].fillna(out[f"{col}_new"])  # fill only where missing
        # Clean helper columns
        drop_cols3 = [c for c in out.columns if c.endswith('_new')]
        out = out.drop(columns=drop_cols3)
        # Finally, enrich odds/lines/weather for ALL rows (including finals) from features
        try:
            feat_keep_all = feat[['game_id','home_team','away_team'] + [c for c in line_cols if c in feat.columns]].drop_duplicates()
            if 'game_id' in out.columns and 'game_id' in feat_keep_all.columns and feat_keep_all['game_id'].notna().any():
                out2 = out.merge(feat_keep_all, on='game_id', how='left', suffixes=('', '_f2'))
            else:
                join_keys2 = [c for c in ['home_team','away_team'] if c in out.columns and c in feat_keep_all.columns]
                out2 = out.merge(feat_keep_all, on=join_keys2, how='left', suffixes=('', '_f2'))
            for col in [c for c in line_cols if c in out2.columns and f"{c}_f2" in out2.columns]:
                out2[col] = out2[col].where(out2[col].notna(), out2[f"{col}_f2"])  # fill only missing
            drop_cols4 = [c for c in out2.columns if c.endswith('_f2')]
            out = out2.drop(columns=drop_cols4)
        except Exception:
            pass
        return out
    except Exception:
        try:
            # If enrichment partially succeeded, prefer returning that
            return out_base  # type: ignore[name-defined]
        except Exception:
            return view_df


def _load_team_assets() -> Dict[str, Dict[str, str]]:
    try:
        with open(ASSETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_stadium_meta_map() -> Dict[str, Dict[str, Any]]:
    """Return mapping of team -> {'stadium': str|None, 'tz': str|None, 'roof': str|None, 'surface': str|None} if available."""
    try:
        import pandas as pd  # already imported at top
        if STADIUM_META_FILE.exists():
            df = pd.read_csv(STADIUM_META_FILE)
            # Normalize team key
            if 'team' in df.columns:
                df['team'] = df['team'].astype(str)
                cols = {c: c for c in ['stadium','tz','roof','surface'] if c in df.columns}
                if cols:
                    return df.set_index('team')[list(cols.keys())].to_dict(orient='index')
    except Exception:
        pass
    return {}


def _file_status(fp: Path) -> Dict[str, Any]:
    try:
        exists = fp.exists()
        size = int(fp.stat().st_size) if exists else 0
        mtime = int(fp.stat().st_mtime) if exists else None
        sha = None
        if exists and size <= 5_000_000:  # avoid hashing very large files
            h = hashlib.sha256()
            with open(fp, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    h.update(chunk)
            sha = h.hexdigest()[:16]
        return {"path": str(fp), "exists": exists, "size": size, "mtime": mtime, "sha256_16": sha}
    except Exception:
        return {"path": str(fp), "exists": False}


@app.route('/api/data-status')
def api_data_status():
    """Report presence and small hashes of key data files to debug parity between local and Render."""
    files = {
        "predictions": PRED_FILE,
        "lines": DATA_DIR / 'lines.csv',
        "games": DATA_DIR / 'games.csv',
        "team_stats": DATA_DIR / 'team_stats.csv',
        "eval_summary": DATA_DIR / 'eval_summary.json',
        "assets": ASSETS_FILE,
        "stadium_meta": STADIUM_META_FILE,
        "overrides": LOCATION_OVERRIDES_FILE,
    }
    status = {k: _file_status(v) for k, v in files.items()}
    # also list a couple of JSON odds snapshots if present
    try:
        candidates = []
        for pat in ('real_betting_lines_*.json','real_betting_lines.json'):
            candidates.extend([p for p in DATA_DIR.glob(pat)])
        status['json_odds'] = [ _file_status(p) for p in sorted(candidates)[:5] ]
    except Exception:
        status['json_odds'] = []
    return jsonify({
        "env": {"RENDER": os.getenv('RENDER'), "DISABLE_JSON_ODDS": os.getenv('DISABLE_JSON_ODDS')},
        "data": status
    })


def _load_location_overrides() -> Dict[str, Dict[str, Any]]:
    """Load optional per-game location overrides.
    CSV columns (any subset): game_id, date, home_team, away_team, venue, city, country, tz, lat, lon, roof, surface, neutral_site
    Returns mapping with two keys:
      - 'by_game_id': {game_id: {...}}
      - 'by_match': {(date, home_team, away_team): {...}}
    """
    out = {"by_game_id": {}, "by_match": {}}
    try:
        if LOCATION_OVERRIDES_FILE.exists():
            expected = ['game_id','date','home_team','away_team','venue','city','country','tz','lat','lon','roof','surface','neutral_site','note']
            # Support commented headers and missing header row
            df = pd.read_csv(LOCATION_OVERRIDES_FILE, comment='#', header=None, names=expected)
            norm = lambda s: None if s is None or (isinstance(s, float) and pd.isna(s)) else str(s)
            for _, r in df.iterrows():
                rec = {k: r.get(k) for k in [
                    'venue','city','country','tz','lat','lon','roof','surface','neutral_site','note'
                ] if k in df.columns}
                gid = norm(r.get('game_id')) if 'game_id' in df.columns else None
                date = norm(r.get('date')) if 'date' in df.columns else None
                home = norm(r.get('home_team')) if 'home_team' in df.columns else None
                away = norm(r.get('away_team')) if 'away_team' in df.columns else None
                if gid:
                    out['by_game_id'][gid] = rec
                if date and home and away:
                    out['by_match'][(date, home, away)] = rec
    except Exception:
        pass
    return out


def _infer_current_season_week(df: pd.DataFrame) -> Optional[tuple[int, int]]:
    """Infer the current (season, week) to display based on game completeness and dates.

    Rules:
    - Respect explicit override via env or nfl_compare/data/current_week.json.
    - Use latest season present in the data.
    - Within that season, pick the earliest week that is not fully completed (any game missing final scores),
      else if scores are unavailable, pick the earliest week whose latest game datetime (max) is in the future.
    - If all weeks are completed and in the past, pick the latest past week.
    This prevents advancing to the next week while the current week still has games remaining.
    """
    try:
        # Global override (env or data/current_week.json)
        ovr = _load_current_week_override()
        if ovr is not None:
            return ovr
        if df is None or df.empty:
            return None
        if not {'season','week'}.issubset(df.columns):
            return None
        # Constrain to latest season
        try:
            latest_season = int(pd.to_numeric(df['season'], errors='coerce').dropna().max())
        except Exception:
            latest_season = None
        if latest_season is None:
            return None
        sdf = df[df['season'] == latest_season].copy()
        if sdf.empty:
            return None
        # Check completion by scores if available
        has_scores = {'home_score','away_score'}.issubset(sdf.columns)
        if has_scores:
            hs = pd.to_numeric(sdf['home_score'], errors='coerce')
            as_ = pd.to_numeric(sdf['away_score'], errors='coerce')
            sdf['_done'] = hs.notna() & as_.notna()
            comp = sdf.groupby('week')['_done'].agg(['sum','count']).reset_index().rename(columns={'sum':'completed','count':'total'})
            # Earliest week with any not-done rows is the active week
            active = comp[comp['completed'] < comp['total']]
            if not active.empty:
                w = int(pd.to_numeric(active['week'], errors='coerce').dropna().min())
                return latest_season, w
        # Fallback to datetime if scores not available or all weeks "complete"
        dt_col = 'game_date' if 'game_date' in sdf.columns else ('date' if 'date' in sdf.columns else None)
        if dt_col:
            sdf[dt_col] = pd.to_datetime(sdf[dt_col], errors='coerce')
            grp = sdf.groupby('week')[dt_col].agg(['min','max']).reset_index().rename(columns={'min':'min_dt','max':'max_dt'})
            now = pd.Timestamp.now()
            pending = grp[grp['max_dt'] >= now]
            if not pending.empty:
                w = int(pd.to_numeric(pending['week'], errors='coerce').dropna().min())
                return latest_season, w
            # Latest past week
            try:
                last = grp.sort_values('max_dt', ascending=True).iloc[-1]
                return latest_season, int(last['week'])
            except Exception:
                pass
        # Last resort: max week seen in latest season
        try:
            wmax = int(pd.to_numeric(sdf['week'], errors='coerce').dropna().max())
            return latest_season, wmax
        except Exception:
            return None
    except Exception:
        return None


# --- Betting EV helpers ---
def _american_to_decimal(ml: Optional[float]) -> Optional[float]:
    try:
        if ml is None or (isinstance(ml, float) and pd.isna(ml)):
            return None
        v = float(ml)
        if v > 0:
            return 1.0 + v / 100.0
        else:
            return 1.0 + 100.0 / abs(v)
    except Exception:
        return None


def _ev_from_prob_and_decimal(p: float, dec_odds: float) -> float:
    """Risk 1 unit. EV in units: p * (dec-1) - (1-p) * 1."""
    win = max(dec_odds - 1.0, 0.0)
    return p * win - (1.0 - p) * 1.0


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 1.0 if x > 0 else 0.0


def _cover_prob_from_edge(edge_pts: float, scale: float) -> float:
    """Map point edge to win probability via logistic with scale parameter."""
    if scale <= 0:
        scale = 1.0
    return _sigmoid(edge_pts / scale)


def _conf_from_ev(ev_units: float) -> Optional[str]:
    """Map EV (in units per 1 risked) to Low/Medium/High; None if not positive.
    Thresholds are percent and can be tuned via env:
      RECS_EV_THRESH_LOW (default 4), RECS_EV_THRESH_MED (8), RECS_EV_THRESH_HIGH (15)
    """
    if ev_units is None or not isinstance(ev_units, (int, float)):
        return None
    if ev_units <= 0:
        return None
    # Read thresholds in percent with safe fallbacks
    try:
        th_low = float(os.environ.get('RECS_EV_THRESH_LOW', '3.5'))
    except Exception:
        th_low = 4.0
    try:
        th_med = float(os.environ.get('RECS_EV_THRESH_MED', '7.5'))
    except Exception:
        th_med = 8.0
    try:
        th_high = float(os.environ.get('RECS_EV_THRESH_HIGH', '12.5'))
    except Exception:
        th_high = 15.0
    ev_pct = ev_units * 100.0
    if ev_pct >= th_high:
        return "High"
    if ev_pct >= th_med:
        return "Medium"
    if ev_pct >= th_low:
        return "Low"
    return None


def _tier_to_num(t: Optional[str]) -> int:
    m = {None: 0, "": 0, "Low": 1, "Medium": 2, "High": 3}
    return m.get(t, 0)


def _num_to_tier(n: int) -> Optional[str]:
    n = max(0, min(3, int(n)))
    r = {0: None, 1: "Low", 2: "Medium", 3: "High"}
    return r[n]


def _combine_confs(*tiers: Optional[str]) -> Optional[str]:
    """Conservatively combine tiers by taking the weakest non-null tier.
    Order of strength: None < Low < Medium < High.
    """
    values = [_tier_to_num(t) for t in tiers if t]
    if not values:
        return None
    return _num_to_tier(min(values))


def _clamp_prob_to_band(p: float, anchor: Optional[float], band: float) -> float:
    try:
        if anchor is None or not isinstance(anchor, (int, float)) or not isinstance(p, (int, float)):
            return p
        lo = max(0.0, float(anchor) - float(band))
        hi = min(1.0, float(anchor) + float(band))
        return max(lo, min(hi, float(p)))
    except Exception:
        return p


def _implied_probs_from_moneylines(ml_home: Optional[float], ml_away: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    """Return de-vig implied win probs from American odds for home and away."""
    try:
        dh = _american_to_decimal(ml_home)
        da = _american_to_decimal(ml_away)
        if dh is None or da is None:
            return (None, None)
        # Convert decimal odds back to implied probs with vig
        # dec = 1 + b/a => implied prob ~ 1/dec
        ph = 1.0 / dh
        pa = 1.0 / da
        s = ph + pa
        if s <= 0:
            return (None, None)
        return (ph / s, pa / s)
    except Exception:
        return (None, None)


def _format_game_datetime(date_like: Any, tz: Optional[str]) -> Optional[str]:
    """Format a date or datetime value to a friendly string like 'Sat, Aug 23, 2025, 11:00 AM TZ'.
    Falls back gracefully if parsing fails or time missing."""
    try:
        if date_like is None or (isinstance(date_like, float) and pd.isna(date_like)):
            return None
        ts = pd.to_datetime(date_like, errors='coerce')
        if pd.isna(ts):
            return str(date_like)
        # If no time component, omit time
        has_time = not (ts.hour == 0 and ts.minute == 0 and ts.second == 0 and getattr(ts, 'nanosecond', 0) == 0)
        if has_time:
            s = ts.strftime('%a, %b %d, %Y, %I:%M %p')
        else:
            s = ts.strftime('%a, %b %d, %Y')
        if tz:
            s = f"{s} {tz}"
        return s
    except Exception:
        return str(date_like) if date_like is not None else None


def _compute_recommendations_for_row(row: pd.Series) -> List[Dict[str, Any]]:
    """Compute EV-based recommendations (ML, Spread, Total) for a single game row.
    Returns a list of recommendation dicts with keys: type, selection, odds, ev_units, ev_pct, confidence, sort_weight, and some game metadata.
    """
    recs: List[Dict[str, Any]] = []

    def g(key: str, *alts: str, default=None):
        """Get first non-null value among provided keys from the row.
        Treats pandas NA/NaT as missing so we properly fall back (e.g., game_date -> date).
        """
        keys = (key, *alts)
        for k in keys:
            if k in row.index:
                v = row.get(k)
                try:
                    if v is None or pd.isna(v):
                        continue
                except Exception:
                    # If pd.isna fails for this type, still accept v as-is
                    pass
                return v
        return default

    home = g("home_team")
    away = g("away_team")
    season = g("season")
    week = g("week")
    game_date = g("game_date", "date")
    # Actuals for grading
    actual_home = g("home_score")
    actual_away = g("away_score")
    actual_total = None
    actual_margin = None
    is_final = False
    if actual_home is not None and actual_away is not None and not pd.isna(actual_home) and not pd.isna(actual_away):
        try:
            ah = float(actual_home); aa = float(actual_away)
            actual_total = ah + aa
            actual_margin = ah - aa
            is_final = True
        except Exception:
            pass

    # Moneyline recommendation logic unified with card view: derive winner via predicted margin, flip probability orientation if inconsistent.
    wp_home = g("pred_home_win_prob", "prob_home_win")
    ml_home = g("moneyline_home")
    ml_away = g("moneyline_away")
    dec_home = _american_to_decimal(ml_home) if ml_home is not None else None
    dec_away = _american_to_decimal(ml_away) if ml_away is not None else None
    try:
        p_home_raw = float(wp_home) if (wp_home is not None and not pd.isna(wp_home)) else None
    except Exception:
        p_home_raw = None
    # Determine margin-based winner (prefer explicit pred_margin if present else compute from predicted points)
    margin_pred = None
    if 'pred_margin' in row.index and not pd.isna(row.get('pred_margin')):
        try:
            margin_pred = float(row.get('pred_margin'))
        except Exception:
            margin_pred = None
    if margin_pred is None:
        ph_tmp = g("pred_home_points", "pred_home_score")
        pa_tmp = g("pred_away_points", "pred_away_score")
        try:
            if ph_tmp is not None and pa_tmp is not None and not pd.isna(ph_tmp) and not pd.isna(pa_tmp):
                margin_pred = float(ph_tmp) - float(pa_tmp)
        except Exception:
            margin_pred = None
    model_winner_by_margin = None
    if margin_pred is not None:
        if margin_pred > 0:
            model_winner_by_margin = home
        elif margin_pred < 0:
            model_winner_by_margin = away
        else:
            model_winner_by_margin = None
    p_home_eff = p_home_raw
    # Flip probability orientation if it disagrees with margin-derived winner
    if p_home_eff is not None and model_winner_by_margin is not None:
        prob_implies_home = (p_home_eff >= 0.5)
        margin_implies_home = (model_winner_by_margin == home)
        if prob_implies_home != margin_implies_home:
            p_home_eff = 1.0 - p_home_eff
    # Compute EV ONLY for margin-based winner (if available); fallback to probability winner if margin missing
    rec_model_winner = model_winner_by_margin
    model_winner_prob = None
    model_winner_ev = None
    if rec_model_winner is not None and p_home_eff is not None:
        if rec_model_winner == home:
            model_winner_prob = p_home_eff
            if dec_home is not None:
                model_winner_ev = _ev_from_prob_and_decimal(model_winner_prob, dec_home)
        else:
            model_winner_prob = 1.0 - p_home_eff
            if dec_away is not None:
                model_winner_ev = _ev_from_prob_and_decimal(model_winner_prob, dec_away)
    elif rec_model_winner is None and p_home_eff is not None:
        # Fallback orientation
        rec_model_winner = home if p_home_eff >= 0.5 else away
        if rec_model_winner == home and dec_home is not None:
            model_winner_prob = p_home_eff
            model_winner_ev = _ev_from_prob_and_decimal(model_winner_prob, dec_home)
        elif rec_model_winner == away and dec_away is not None:
            model_winner_prob = 1.0 - p_home_eff
            model_winner_ev = _ev_from_prob_and_decimal(model_winner_prob, dec_away)
    # Grade and append if positive EV
    if model_winner_ev is not None and model_winner_ev > 0:
        conf = _conf_from_ev(model_winner_ev)
        ml_result = None
        if is_final and actual_margin is not None:
            if actual_margin == 0:
                ml_result = "Push"
            else:
                actual_winner_team = home if actual_margin > 0 else away
                ml_result = "Win" if actual_winner_team == rec_model_winner else "Loss"
        recs.append({
            "type": "MONEYLINE",
            "selection": f"{rec_model_winner} ML",
            "odds": int(ml_home if rec_model_winner==home else ml_away) if isinstance(ml_home if rec_model_winner==home else ml_away, (int,float)) else (ml_home if rec_model_winner==home else ml_away),
            "ev_units": model_winner_ev,
            "ev_pct": model_winner_ev * 100.0,
            "confidence": conf,
            "sort_weight": (_tier_to_num(conf), model_winner_ev or -999),
            "season": season, "week": week, "game_date": game_date, "home_team": home, "away_team": away,
            "result": ml_result,
        })

    # Spread (ATS) at -110
    margin = None
    # Prefer close spread for finals; fallback otherwise
    _hs = g("home_score"); _as = g("away_score")
    _is_final = (_hs is not None and not pd.isna(_hs)) and (_as is not None and not pd.isna(_as))
    spread = g("close_spread_home") if _is_final else g("market_spread_home", "spread_home", "open_spread_home")
    if spread is None or (isinstance(spread, float) and pd.isna(spread)):
        spread = g("market_spread_home", "spread_home", "open_spread_home")
    try:
        ph = g("pred_home_points", "pred_home_score")
        pa = g("pred_away_points", "pred_away_score")
        if ph is not None and pa is not None and not pd.isna(ph) and not pd.isna(pa):
            margin = float(ph) - float(pa)
    except Exception:
        margin = None
    if margin is not None and spread is not None and not pd.isna(spread):
        try:
            edge_pts = float(margin) + float(spread)
            scale_margin = float(os.environ.get('NFL_ATS_SIGMA', '9.0'))
        except Exception:
            edge_pts, scale_margin = None, 9.0
        if edge_pts is not None:
            p_home_cover = _cover_prob_from_edge(edge_pts, scale_margin)
            try:
                shrink = float(os.environ.get('RECS_PROB_SHRINK', '0.35'))
            except Exception:
                shrink = 0.35
            p_home_cover = 0.5 + (p_home_cover - 0.5) * (1.0 - shrink)
            # Use market prices if available; fallback to -110
            sh_price = g("spread_home_price")
            sa_price = g("spread_away_price")
            dec_home_sp = _american_to_decimal(sh_price) if sh_price is not None and not pd.isna(sh_price) else 1.0 + 100.0/110.0
            dec_away_sp = _american_to_decimal(sa_price) if sa_price is not None and not pd.isna(sa_price) else 1.0 + 100.0/110.0
            ev_home = _ev_from_prob_and_decimal(p_home_cover, dec_home_sp)
            ev_away = _ev_from_prob_and_decimal(1.0 - p_home_cover, dec_away_sp)
            # Build selections
            try:
                sp = float(spread)
            except Exception:
                sp = None
            # Normalize price display as signed ints if present
            sh_disp = (int(sh_price) if (sh_price is not None and not pd.isna(sh_price)) else None)
            sa_disp = (int(sa_price) if (sa_price is not None and not pd.isna(sa_price)) else None)
            if sp is not None:
                home_sel = f"{home} {sp:+.1f}{(' ('+('%+d' % sh_disp)+')') if (sh_disp is not None) else ''}"
                away_sel = f"{away} {(-sp):+.1f}{(' ('+('%+d' % sa_disp)+')') if (sa_disp is not None) else ''}"
            else:
                home_sel = f"{home} ATS{(' ('+('%+d' % sh_disp)+')') if (sh_disp is not None) else ''}"
                away_sel = f"{away} ATS{(' ('+('%+d' % sa_disp)+')') if (sa_disp is not None) else ''}"
            # Choose best
            cand = [(home_sel, ev_home), (away_sel, ev_away)]
            cand = [(s, e) for s, e in cand if e is not None]
            if cand:
                s, e = max(cand, key=lambda t: t[1])
                # Confidence is based on this market's EV only
                conf = _conf_from_ev(e)
                # Grade if final
                result = None
                if is_final and actual_margin is not None and sp is not None:
                    cover_val = actual_margin + float(spread)
                    actual_side = "HOME" if cover_val > 0 else ("AWAY" if cover_val < 0 else "PUSH")
                    picked_side = "HOME" if str(s).startswith(str(home)) else ("AWAY" if str(s).startswith(str(away)) else None)
                    if picked_side:
                        if actual_side == "PUSH":
                            result = "Push"
                        else:
                            result = "Win" if picked_side == actual_side else "Loss"
                recs.append({
                    "type": "SPREAD",
                    "selection": s,
                    "odds": -110,
                    "ev_units": e,
                    "ev_pct": e * 100.0 if e is not None else None,
                    "confidence": conf,
                    "sort_weight": (_tier_to_num(conf), e or -999),
                    "season": season, "week": week, "game_date": game_date, "home_team": home, "away_team": away,
                    "result": result,
                })

    # Total at -110
    total_pred = g("pred_total")
    m_total = g("close_total") if _is_final else g("market_total", "total", "open_total")
    if m_total is None or (isinstance(m_total, float) and pd.isna(m_total)):
        m_total = g("market_total", "total", "open_total")
    if total_pred is not None and not pd.isna(total_pred) and m_total is not None and not pd.isna(m_total):
        try:
            edge_t = float(total_pred) - float(m_total)
            scale_total = float(os.environ.get('NFL_TOTAL_SIGMA', '10.0'))
        except Exception:
            edge_t, scale_total = None, 10.0
        if edge_t is not None:
            p_over = _cover_prob_from_edge(edge_t, scale_total)
            try:
                shrink = float(os.environ.get('RECS_PROB_SHRINK', '0.35'))
            except Exception:
                shrink = 0.35
            p_over = 0.5 + (p_over - 0.5) * (1.0 - shrink)
            # Use market prices if available; fallback to -110
            to_price = g("total_over_price")
            tu_price = g("total_under_price")
            dec_over = _american_to_decimal(to_price) if to_price is not None and not pd.isna(to_price) else 1.0 + 100.0/110.0
            dec_under = _american_to_decimal(tu_price) if tu_price is not None and not pd.isna(tu_price) else 1.0 + 100.0/110.0
            ev_over = _ev_from_prob_and_decimal(p_over, dec_over)
            ev_under = _ev_from_prob_and_decimal(1.0 - p_over, dec_under)
            # Choose best
            try:
                tot = float(m_total)
            except Exception:
                tot = None
            to_disp = (int(to_price) if (to_price is not None and not pd.isna(to_price)) else None)
            tu_disp = (int(tu_price) if (tu_price is not None and not pd.isna(tu_price)) else None)
            over_sel = (f"Over {tot:.1f}" if tot is not None else "Over") + (f" ({to_disp:+d})" if (to_disp is not None) else "")
            under_sel = (f"Under {tot:.1f}" if tot is not None else "Under") + (f" ({tu_disp:+d})" if (tu_disp is not None) else "")
            cand = [(over_sel, ev_over), (under_sel, ev_under)]
            cand = [(s, e) for s, e in cand if e is not None]
            if cand:
                s, e = max(cand, key=lambda t: t[1])
                # Confidence is based on this market's EV only
                conf = _conf_from_ev(e)
                # Grade if final
                result = None
                if is_final and actual_total is not None and m_total is not None and not pd.isna(m_total):
                    try:
                        mt = float(m_total)
                        if actual_total > mt:
                            actual_ou = "OVER"
                        elif actual_total < mt:
                            actual_ou = "UNDER"
                        else:
                            actual_ou = "PUSH"
                        picked_ou = "OVER" if str(s).startswith("Over") else ("UNDER" if str(s).startswith("Under") else None)
                        if picked_ou:
                            if actual_ou == "PUSH":
                                result = "Push"
                            else:
                                result = "Win" if picked_ou == actual_ou else "Loss"
                    except Exception:
                        result = None
                recs.append({
                    "type": "TOTAL",
                    "selection": s,
                    "odds": -110,
                    "ev_units": e,
                    "ev_pct": e * 100.0 if e is not None else None,
                    "confidence": conf,
                    "sort_weight": (_tier_to_num(conf), e or -999),
                    "season": season, "week": week, "game_date": game_date, "home_team": home, "away_team": away,
                    "result": result,
                })

    # Apply global filtering to reduce noise
    try:
        # Default minimum EV (2%) so Week 1 isn't empty; can be overridden via query or env
        min_ev_pct = float(os.environ.get('RECS_MIN_EV_PCT', '2.0'))
    except Exception:
        min_ev_pct = 2.0
    include_completed = str(os.environ.get('RECS_INCLUDE_COMPLETED', 'true')).strip().lower() in {'1','true','yes','y'}
    filtered: List[Dict[str, Any]] = []
    for r in recs:
        evp = r.get('ev_pct')
        if evp is not None and evp >= min_ev_pct:
            filtered.append(r)
        elif is_final and include_completed:
            filtered.append(r)
    # Ensure every passing pick has a visible confidence tier; only floor when none computed
    for r in filtered:
        if (r.get('confidence') is None or r.get('confidence') == '') and r.get('ev_pct') is not None and r.get('ev_pct') >= min_ev_pct:
            r['confidence'] = 'Low'
            w_ev = r.get('ev_units') or -999
            r['sort_weight'] = (_tier_to_num('Low'), w_ev)
    # Optional: keep only the single highest-EV recommendation per game
    one_per_game = str(os.environ.get('RECS_ONE_PER_GAME', 'false')).strip().lower() in {'1','true','yes','y'}
    if one_per_game and filtered:
        best = max(filtered, key=lambda r: r.get('ev_units') if r.get('ev_units') is not None else -999)
        return [best]
    return filtered


@app.route("/health")
def health():
    return {"status": "ok", "have_predictions": PRED_FILE.exists()}, 200


@app.route('/favicon.ico')
def favicon():
    # Return a 1x1 transparent PNG to avoid 404s from browsers requesting /favicon.ico
    import base64
    png_b64 = (
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y6r/RwAAAAASUVORK5CYII="
    )
    png = base64.b64decode(png_b64)
    from flask import Response
    return Response(png, mimetype='image/png')


@app.route("/api/player-props")
def api_player_props():
    """Compute and return weekly player prop projections.

    Query params:
      - season (int): season year; defaults to latest season present in games/predictions.
      - week (int): week number; defaults to inferred current week (or 1).
      - position (str): optional filter like QB,RB,WR,TE,DEF
      - team (str): team abbr/name (normalized).
    """
    # Note: We defer importing compute_player_props until we actually need to compute,
    # so the endpoint can still serve cached data when heavy dependencies are unavailable.

    pred_df = _load_predictions()
    games_df = _load_games()
    # Respect deployment flag to avoid heavy on-request computations (especially on Render free plan)
    disable_on_request = str(os.environ.get('DISABLE_ON_REQUEST_PREDICTIONS', '1')).strip().lower() in {'1','true','yes','y'}

    # Determine season/week defaults similar to other endpoints
    season = request.args.get("season")
    week = request.args.get("week")
    season_i = None
    week_i = None
    try:
        season_i = int(season) if season else None
    except Exception:
        season_i = None
    try:
        week_i = int(week) if week else None
    except Exception:
        week_i = None
    if week_i is None:
        # Infer current week from games/predictions
        try:
            src = games_df if (games_df is not None and not games_df.empty) else pred_df
            inferred = _infer_current_season_week(src) if (src is not None and not src.empty) else None
            if inferred is not None:
                season_i, week_i = int(inferred[0]), int(inferred[1])
            else:
                if season_i is None and src is not None and not src.empty and 'season' in src.columns and not src['season'].isna().all():
                    season_i = int(src['season'].max())
                week_i = 1
        except Exception:
            week_i = 1
    if season_i is None:
        try:
            src = games_df if (games_df is not None and not games_df.empty) else pred_df
            if src is not None and not src.empty and 'season' in src.columns and not src['season'].isna().all():
                season_i = int(src['season'].max())
        except Exception:
            season_i = None

    if season_i is None or week_i is None:
        return jsonify({"rows": 0, "data": [], "note": "season/week could not be determined"})

    # Fast path: use precomputed CSV if available (compute only when missing or forced)
    force = (request.args.get('force', '0').lower() in {'1','true','yes'}) and (not disable_on_request)
    cache_fp = DATA_DIR / f"player_props_{season_i}_wk{week_i}.csv"
    df = None
    if not force and cache_fp.exists():
        try:
            df = pd.read_csv(cache_fp)
        except Exception:
            df = None
    if (df is None or df.empty) and disable_on_request:
        # On-demand compute disabled; attempt to serve the latest available cache instead of empty
        fallback = _find_latest_props_cache(prefer_season=season_i, prefer_week=week_i)
        if fallback is not None:
            fp, s_fb, w_fb = fallback
            try:
                df = pd.read_csv(fp)
                season_i, week_i = s_fb, w_fb
            except Exception:
                df = None
        if df is None or df.empty:
            return jsonify({"rows": 0, "data": [], "season": season_i, "week": week_i, "note": "on-demand props computation disabled; generate cache offline"})
    if df is None or df.empty:
        try:
            from nfl_compare.src.player_props import compute_player_props  # defer import
            df = compute_player_props(season_i, week_i)
            # Persist for subsequent fast loads
            try:
                df.to_csv(cache_fp, index=False)
            except Exception:
                pass
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    # Sanity check: if cache contained stale/malformed rows (e.g., QB rushing NaN), recompute and refresh cache
    try:
        if df is not None and not df.empty and 'position' in df.columns:
            pos_up = df['position'].astype(str).str.upper()
            qb_mask = (pos_up == 'QB')
            needs_fix = False
            if qb_mask.any():
                # If any QB has pass_attempts present but missing rush fields, it's likely an old cache
                if 'pass_attempts' in df.columns and 'rush_attempts' in df.columns and 'rush_yards' in df.columns:
                    pa_ok = pd.to_numeric(df.loc[qb_mask, 'pass_attempts'], errors='coerce').notna()
                    ra_nan = pd.to_numeric(df.loc[qb_mask, 'rush_attempts'], errors='coerce').isna()
                    ry_nan = pd.to_numeric(df.loc[qb_mask, 'rush_yards'], errors='coerce').isna()
                    if (pa_ok & (ra_nan | ry_nan)).any():
                        needs_fix = True
            if needs_fix:
                try:
                    from nfl_compare.src.player_props import compute_player_props  # defer import
                    fresh = compute_player_props(season_i, week_i)
                    if fresh is not None and not fresh.empty:
                        df = fresh
                        try:
                            df.to_csv(cache_fp, index=False)
                        except Exception:
                            pass
                except Exception:
                    # If recompute fails, fall back to existing df
                    pass
    except Exception:
        pass
    if df is None or df.empty:
        return jsonify({"rows": 0, "data": []})

    # Defensive sanitation: coerce numeric projections and backfill obvious missing fields
    fixes_applied = []
    try:
        # Coerce common numeric columns to numbers (malformed caches may load as object)
        num_like = [
            'pass_attempts','pass_yards','pass_tds','interceptions',
            'rush_attempts','rush_yards','rush_tds',
            'targets','receptions','rec_yards','rec_tds',
            'any_td_prob'
        ]
        for c in [c for c in num_like if c in df.columns]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        # Backfill receiving yards/receptions when targets present but yards/receptions missing
        if {'position','targets'}.issubset(df.columns):
            pos_up = df['position'].astype(str).str.upper()
            # Position heuristics (conservative baselines)
            pos_ypt = {'WR': 8.3, 'TE': 7.6, 'RB': 6.4}
            pos_cr = {'WR': 0.62, 'TE': 0.66, 'RB': 0.72}
            # rec_yards
            if 'rec_yards' in df.columns:
                m = df['rec_yards'].isna() & df['targets'].notna() & (pos_up.isin(['WR','TE','RB']))
                if m.any():
                    est = df.loc[m, 'targets'].astype(float) * df.loc[m, 'position'].astype(str).str.upper().map(pos_ypt).fillna(7.8)
                    df.loc[m, 'rec_yards'] = est
                    fixes_applied.append(f"backfilled rec_yards for {int(m.sum())} rows")
            # receptions
            if 'receptions' in df.columns:
                m2 = df['receptions'].isna() & df['targets'].notna() & (pos_up.isin(['WR','TE','RB']))
                if m2.any():
                    est2 = df.loc[m2, 'targets'].astype(float) * df.loc[m2, 'position'].astype(str).str.upper().map(pos_cr).fillna(0.64)
                    df.loc[m2, 'receptions'] = est2
                    fixes_applied.append(f"backfilled receptions for {int(m2.sum())} rows")
        # Backfill QB rush_yards if attempts present but yards missing
        if {'position','rush_attempts','rush_yards'}.issubset(df.columns):
            qb = df['position'].astype(str).str.upper().eq('QB')
            m3 = qb & df['rush_yards'].isna() & df['rush_attempts'].notna()
            if m3.any():
                df.loc[m3, 'rush_yards'] = df.loc[m3, 'rush_attempts'].astype(float) * 3.5  # conservative YPC
                fixes_applied.append(f"backfilled QB rush_yards for {int(m3.sum())} rows")
    except Exception:
        pass

    # Optional filters
    pos = (request.args.get("position") or "").strip().upper()
    offense_only = (request.args.get('offense', '1').lower() in {'1','true','yes'})
    primary_qb_only = (request.args.get('primary_qb_only', '1').lower() in {'1','true','yes'})
    active_only = (request.args.get('active_only', '1').lower() in {'1','true','yes'})
    team = (request.args.get("team") or "").strip()
    if pos and 'position' in df.columns:
        df = df[df['position'].astype(str).str.upper() == pos]
    elif offense_only and 'position' in df.columns:
        try:
            df = df[df['position'].astype(str).str.upper().isin(['QB','RB','WR','TE'])]
        except Exception:
            pass
    # Filter to the single primary QB (the one with pass_attempts assigned) when requested
    if primary_qb_only and 'position' in df.columns and 'pass_attempts' in df.columns:
        try:
            m = (df['position'].astype(str).str.upper() != 'QB') | (pd.to_numeric(df['pass_attempts'], errors='coerce').notna())
            df = df[m]
        except Exception:
            pass
    if team and 'team' in df.columns:
        # Normalize simple variants (we don't import normalizer here to keep light)
        df = df[df['team'].astype(str).str.lower() == team.lower()]
    # Filter inactives by default if the column exists; if not present and requested, recompute fresh to attach it
    if active_only:
        try:
            if 'is_active' not in df.columns:
                try:
                    fresh = compute_player_props(season_i, week_i)
                    if fresh is not None and not fresh.empty:
                        df = fresh
                        try:
                            df.to_csv(cache_fp, index=False)
                        except Exception:
                            pass
                except Exception:
                    pass
            if 'is_active' in df.columns:
                df = df[pd.to_numeric(df['is_active'], errors='coerce').fillna(1).astype(int) == 1]
        except Exception:
            pass

    # Apply display aliases (presentation only)
    try:
        df = _apply_display_aliases(df)
    except Exception:
        pass
    # Round numeric columns to 2 decimals for consistency in API responses
    try:
        num_cols = df.select_dtypes(include='number').columns
        if len(num_cols) > 0:
            df[num_cols] = df[num_cols].round(2)
    except Exception:
        pass
    # Ensure valid JSON: replace NaN/NaT with None so the response doesn't contain NaN tokens
    try:
        df = df.where(pd.notnull(df), None)
    except Exception:
        pass

    # Limit columns for API to the key projection fields
    keep = [c for c in [
        'season','week','date','game_id','team','opponent','is_home','player','position',
        'pass_attempts','pass_yards','pass_tds','interceptions',
        'rush_attempts','rush_yards','rush_tds',
        'targets','receptions','rec_yards','rec_tds',
        'tackles','sacks',
        'any_td_prob',
    ] if c in df.columns]
    data = df[keep].to_dict(orient='records') if keep else df.to_dict(orient='records')

    # JSON-safe: recursively replace NaN/Inf with None so fetch().json() doesn't choke in browsers
    def _json_safe(obj):
        if isinstance(obj, float):
            try:
                if math.isfinite(obj):
                    return obj
                return None
            except Exception:
                return obj
        if obj is None:
            return None
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(x) for x in obj]
        return obj

    data = _json_safe(data)
    resp = {"rows": len(data), "data": data, "season": season_i, "week": week_i}
    # If we served a fallback cache earlier in the function, include a note.
    # Note: variable may not exist if path didn't take fallback branch; guard via locals().
    try:
        fb = locals().get('fallback')
        if fb is not None:
            _, s_fb, w_fb = fb
            resp["note"] = f"served fallback cache: season={s_fb}, week={w_fb}"
    except Exception:
        pass
    # Attach sanitation note if any fixes were applied
    try:
        if fixes_applied:
            note_str = "; ".join(fixes_applied)
            if 'note' in resp and resp['note']:
                resp['note'] = f"{resp['note']} | {note_str}"
            else:
                resp['note'] = note_str
    except Exception:
        pass
    return jsonify(resp)


@app.route("/api/player-props.csv")
def api_player_props_csv():
    """Return weekly player prop projections as CSV (downloadable).

    Query params:
      - season (int)
      - week (int)
      - position (str)
      - team (str)
    """
    # Defer importing compute_player_props until computation is necessary.

    pred_df = _load_predictions()
    games_df = _load_games()
    disable_on_request = str(os.environ.get('DISABLE_ON_REQUEST_PREDICTIONS', '1')).strip().lower() in {'1','true','yes','y'}

    season = request.args.get("season")
    week = request.args.get("week")
    try:
        season_i = int(season) if season else None
    except Exception:
        season_i = None
    try:
        week_i = int(week) if week else None
    except Exception:
        week_i = None
    if week_i is None:
        try:
            src = games_df if (games_df is not None and not games_df.empty) else pred_df
            inferred = _infer_current_season_week(src) if (src is not None and not src.empty) else None
            if inferred is not None:
                season_i, week_i = int(inferred[0]), int(inferred[1])
            else:
                if season_i is None and src is not None and not src.empty and 'season' in src.columns and not src['season'].isna().all():
                    season_i = int(src['season'].max())
                week_i = 1
        except Exception:
            week_i = 1
    if season_i is None:
        try:
            src = games_df if (games_df is not None and not games_df.empty) else pred_df
            if src is not None and not src.empty and 'season' in src.columns and not src['season'].isna().all():
                season_i = int(src['season'].max())
        except Exception:
            season_i = None

    if season_i is None or week_i is None:
        return jsonify({"rows": 0, "data": [], "note": "season/week could not be determined"})

    # Fast path: use precomputed CSV if available (compute only when missing or forced)
    force = (request.args.get('force', '0').lower() in {'1','true','yes'}) and (not disable_on_request)
    cache_fp = DATA_DIR / f"player_props_{season_i}_wk{week_i}.csv"
    df = None
    if not force and cache_fp.exists():
        try:
            df = pd.read_csv(cache_fp)
        except Exception:
            df = None
    if (df is None or df.empty) and disable_on_request:
        # On-demand compute disabled; serve latest available cache if possible
        fallback = _find_latest_props_cache(prefer_season=season_i, prefer_week=week_i)
        if fallback is not None:
            fp, s_fb, w_fb = fallback
            try:
                df = pd.read_csv(fp)
                season_i, week_i = s_fb, w_fb
            except Exception:
                df = None
        if df is None or df.empty:
            try:
                from flask import Response
                empty_csv = "season,week,date,game_id,team,opponent,is_home,player,position,pass_attempts,pass_yards,pass_tds,interceptions,rush_attempts,rush_yards,rush_tds,targets,receptions,rec_yards,rec_tds,tackles,sacks,any_td_prob\n"
                headers = {'Content-Disposition': f'attachment; filename="player_props_{season_i}_wk{week_i}.csv"'}
                return Response(empty_csv, mimetype='text/csv', headers=headers)
            except Exception:
                return jsonify({"rows": 0, "data": [], "season": season_i, "week": week_i, "note": "on-demand props computation disabled; generate cache offline"})
    if df is None or df.empty:
        try:
            from nfl_compare.src.player_props import compute_player_props  # defer import
            df = compute_player_props(season_i, week_i)
            try:
                df.to_csv(cache_fp, index=False)
            except Exception:
                pass
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    if df is None or df.empty:
        return jsonify({"rows": 0, "data": []})

    # Optional filters
    pos = (request.args.get("position") or "").strip().upper()
    offense_only = (request.args.get('offense', '1').lower() in {'1','true','yes'})
    primary_qb_only = (request.args.get('primary_qb_only', '1').lower() in {'1','true','yes'})
    active_only = (request.args.get('active_only', '1').lower() in {'1','true','yes'})
    team = (request.args.get("team") or "").strip()
    if pos and 'position' in df.columns:
        df = df[df['position'].astype(str).str.upper() == pos]
    elif offense_only and 'position' in df.columns:
        try:
            df = df[df['position'].astype(str).str.upper().isin(['QB','RB','WR','TE'])]
        except Exception:
            pass
    if primary_qb_only and 'position' in df.columns and 'pass_attempts' in df.columns:
        try:
            m = (df['position'].astype(str).str.upper() != 'QB') | (pd.to_numeric(df['pass_attempts'], errors='coerce').notna())
            df = df[m]
        except Exception:
            pass
    if team and 'team' in df.columns:
        df = df[df['team'].astype(str).str.lower() == team.lower()]
    # Filter inactives by default
    if active_only:
        try:
            if 'is_active' not in df.columns:
                try:
                    from nfl_compare.src.player_props import compute_player_props  # defer import
                    fresh = compute_player_props(season_i, week_i)
                    if fresh is not None and not fresh.empty:
                        df = fresh
                        try:
                            df.to_csv(cache_fp, index=False)
                        except Exception:
                            pass
                except Exception:
                    pass
            if 'is_active' in df.columns:
                df = df[pd.to_numeric(df['is_active'], errors='coerce').fillna(1).astype(int) == 1]
        except Exception:
            pass

    # Apply display aliases for CSV as well
    try:
        df = _apply_display_aliases(df)
    except Exception:
        pass
    # Stream CSV
    try:
        # Round numeric and format floats with two decimals
        try:
            num_cols = df.select_dtypes(include='number').columns
            if len(num_cols) > 0:
                df[num_cols] = df[num_cols].astype(float).round(2)
        except Exception:
            pass
        from flask import Response
        csv_bytes = df.to_csv(index=False, float_format='%.2f')
        fname = f"player_props_{season_i}_wk{week_i}.csv"
        headers = {
            'Content-Disposition': f'attachment; filename="{fname}"'
        }
        return Response(csv_bytes, mimetype='text/csv', headers=headers)
    except Exception as e:
        return jsonify({"error": f"csv export failed: {e}"}), 500


@app.route("/api/player-props-reconciliation")
def api_player_props_reconciliation():
    """Return reconciliation of projections vs actuals for a given season/week (JSON).

    Query params:
      - season (int, required)
      - week (int, required)
      - position/team filters as in props endpoints (optional)
    """
    try:
        from nfl_compare.src.reconciliation import reconcile_props, summarize_errors  # lazy import
    except Exception as e:
        return jsonify({"error": f"reconciliation unavailable: {e}"}), 500

    # Season/week required to avoid guessing wrong after slate
    try:
        season_i = int(request.args.get("season"))
        week_i = int(request.args.get("week"))
    except Exception:
        return jsonify({"error": "season and week are required"}), 400
    # Cache key and optional refresh control
    cache_key = (season_i, week_i)
    force_refresh = str(request.args.get("refresh") or request.args.get("force") or "0").lower() in {"1","true","yes","y"}
    cache_hit = False
    recon_source = ""
    # Prefer a precomputed CSV cache if present (helps production where heavy compute is disabled)
    try:
        cache_csv_fp = DATA_DIR / f"player_props_vs_actuals_{season_i}_wk{week_i}.csv"
    except Exception:
        cache_csv_fp = None  # type: ignore
    try:
        df = None
        if not force_refresh and cache_csv_fp is not None and cache_csv_fp.exists():
            try:
                df = pd.read_csv(cache_csv_fp)
                recon_source = "cache_csv"
            except Exception:
                df = None
        if df is None:
            df, cache_hit = _recon_cache_get(cache_key, force_refresh=force_refresh)
            if df is not None:
                recon_source = "cache_mem"
        if df is None:
            # Compute via library (may rely on local parquet or nfl-data-py)
            df = reconcile_props(season_i, week_i)
            recon_source = "computed"
            _recon_cache_put(cache_key, df)
            # Persist a CSV cache for production servers
            try:
                if cache_csv_fp is not None:
                    df.to_csv(cache_csv_fp, index=False)
            except Exception:
                pass
    except FileNotFoundError as e:
        # Missing projections file or similar: prefer empty payload with note
        return jsonify({"rows": 0, "data": [], "season": season_i, "week": week_i, "note": str(e)}), 200
    except RuntimeError as e:
        # Likely no weekly actuals available in production; return empty set with note instead of 500
        return jsonify({"rows": 0, "data": [], "season": season_i, "week": week_i, "note": str(e)}), 200
    except Exception as e:
        # Internal unexpected error
        return jsonify({"error": str(e)}), 500
    if df is None or df.empty:
        return jsonify({"rows": 0, "data": [], "season": season_i, "week": week_i})

    # Optional filters
    pos = (request.args.get("position") or "").strip().upper()
    team = (request.args.get("team") or "").strip()
    if pos and 'position' in df.columns:
        df = df[df['position'].astype(str).str.upper() == pos]
    if team and 'team' in df.columns:
        df = df[df['team'].astype(str).str.lower() == team.lower()]

    # Round numeric columns to 2 decimals for consistency in API responses
    try:
        num_cols = df.select_dtypes(include='number').columns
        if len(num_cols) > 0:
            df[num_cols] = df[num_cols].round(2)
    except Exception:
        pass
    # Clean NaNs for JSON and ensure strict JSON safety (no NaN/Inf)
    try:
        df = df.where(pd.notnull(df), None)
    except Exception:
        pass
    data = df.to_dict(orient='records')
    def _json_safe(obj):
        try:
            import math as _m
            if isinstance(obj, float):
                return obj if _m.isfinite(obj) else None
            if obj is None:
                return None
            if isinstance(obj, dict):
                return {k: _json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_json_safe(x) for x in obj]
            return obj
        except Exception:
            return obj
    data = _json_safe(data)
    # Summary
    try:
        summ = summarize_errors(pd.DataFrame(data))
        summ = summ.where(pd.notnull(summ), None)
        summary = summ.to_dict(orient='records') if summ is not None and not summ.empty else []
    except Exception:
        summary = []
    summary = _json_safe(summary)
    from flask import make_response
    resp = make_response(jsonify({"rows": len(data), "data": data, "summary": summary, "season": season_i, "week": week_i}))
    try:
        resp.headers['X-Recon-Cache'] = 'hit' if cache_hit else 'miss'
    except Exception:
        pass
    try:
        if recon_source:
            resp.headers['X-Recon-Source'] = recon_source
    except Exception:
        pass
    return resp


@app.route("/api/player-props-reconciliation.csv")
def api_player_props_reconciliation_csv():
    try:
        from nfl_compare.src.reconciliation import reconcile_props  # lazy import
    except Exception as e:
        return jsonify({"error": f"reconciliation unavailable: {e}"}), 500
    try:
        season_i = int(request.args.get("season"))
        week_i = int(request.args.get("week"))
    except Exception:
        return jsonify({"error": "season and week are required"}), 400
    # Use cache if available
    cache_key = (season_i, week_i)
    force_refresh = str(request.args.get("refresh") or request.args.get("force") or "0").lower() in {"1","true","yes","y"}
    cache_hit = False
    recon_source = ""
    try:
        df = None
        cache_csv_fp = DATA_DIR / f"player_props_vs_actuals_{season_i}_wk{week_i}.csv"
        if not force_refresh and cache_csv_fp.exists():
            try:
                df = pd.read_csv(cache_csv_fp)
                recon_source = "cache_csv"
            except Exception:
                df = None
        if df is None:
            df, cache_hit = _recon_cache_get(cache_key, force_refresh=force_refresh)
            if df is not None:
                recon_source = "cache_mem"
        if df is None:
            df = reconcile_props(season_i, week_i)
            recon_source = "computed"
            _recon_cache_put(cache_key, df)
            try:
                cache_csv_fp.to_csv(cache_csv_fp, index=False)  # type: ignore[attr-defined]
            except Exception:
                # Fallback correct write
                try:
                    df.to_csv(cache_csv_fp, index=False)
                except Exception:
                    pass
    except FileNotFoundError as e:
        # Return an empty CSV with a hint header instead of 404 to keep UI happy
        from flask import Response
        resp = Response("", mimetype='text/csv')
        try:
            resp.headers['X-Recon-Note'] = str(e)
            resp.headers['X-Recon-Source'] = 'missing'
        except Exception:
            pass
        return resp
    except RuntimeError as e:
        # No weekly actuals in production; return empty CSV
        from flask import Response
        resp = Response("", mimetype='text/csv')
        try:
            resp.headers['X-Recon-Note'] = str(e)
            resp.headers['X-Recon-Source'] = 'missing'
        except Exception:
            pass
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    if df is None or df.empty:
        from flask import Response
        # Return a valid but empty CSV
        resp = Response("", mimetype='text/csv')
        try:
            resp.headers['X-Recon-Cache'] = 'hit' if cache_hit else 'miss'
            resp.headers['X-Recon-Source'] = recon_source or 'unknown'
        except Exception:
            pass
        return resp
    # Optional filters (mirror JSON endpoint)
    pos = (request.args.get("position") or "").strip().upper()
    team = (request.args.get("team") or "").strip()
    try:
        if pos and 'position' in df.columns:
            df = df[df['position'].astype(str).str.upper() == pos]
        if team and 'team' in df.columns:
            df = df[df['team'].astype(str).str.lower() == team.lower()]
    except Exception:
        pass
    try:
        # Ensure numeric columns are floats and rounded to 2 decimals, then format as %.2f in CSV
        try:
            num_cols = df.select_dtypes(include='number').columns
            if len(num_cols) > 0:
                df[num_cols] = df[num_cols].astype(float).round(2)
        except Exception:
            pass
        from flask import Response
        csv_bytes = df.to_csv(index=False, float_format='%.2f')
        fname = f"player_props_vs_actuals_{season_i}_wk{week_i}.csv"
        headers = {'Content-Disposition': f'attachment; filename="{fname}"'}
        from flask import make_response
        resp = Response(csv_bytes, mimetype='text/csv', headers=headers)
        try:
            resp.headers['X-Recon-Cache'] = 'hit' if cache_hit else 'miss'
        except Exception:
            pass
        try:
            if recon_source:
                resp.headers['X-Recon-Source'] = recon_source
        except Exception:
            pass
        return resp
    except Exception as e:
        return jsonify({"error": f"csv export failed: {e}"}), 500


@app.route("/api/predictions")
def api_predictions():
    df = _load_predictions()
    if df.empty:
        return {"rows": 0, "data": []}, 200

    # Optional filters
    season = request.args.get("season")
    week = request.args.get("week")
    # Default: latest season, week 1 when no filters provided
    if not season and not week:
        try:
            if "season" in df.columns and not df["season"].isna().all():
                latest_season = int(df["season"].max())
                df = df[df["season"] == latest_season]
        except Exception:
            pass
        if "week" in df.columns:
            df = df[df["week"].astype(str) == "1"]
    if season:
        try:
            season_i = int(season)
            if "season" in df.columns:
                df = df[df["season"] == season_i]
        except ValueError:
            pass
    if week:
        try:
            week_i = int(week)
            if "week" in df.columns:
                df = df[df["week"] == week_i]
        except ValueError:
            pass

    # Limit columns for API clarity if present
    prefer_cols = [
        "season", "week", "game_id", "game_date", "home_team", "away_team",
        "pred_home_points", "pred_away_points", "pred_total", "pred_home_win_prob",
        "market_spread_home", "market_total",
    ]
    cols = [c for c in prefer_cols if c in df.columns]
    out = df[cols].to_dict(orient="records") if cols else df.to_dict(orient="records")
    return {"rows": len(out), "data": out}, 200


@app.route("/api/recommendations")
def api_recommendations():
    """Return EV-based betting recommendations aggregated across games.
    Optional query params: season, week, date (YYYY-MM-DD)
    """
    pred_df = _load_predictions()
    games_df = _load_games()

    # Parse filters
    season = request.args.get("season")
    week = request.args.get("week")
    date = request.args.get("date")
    season_i = None
    week_i = None
    if season:
        try:
            season_i = int(season)
        except Exception:
            season_i = None
    if week:
        try:
            week_i = int(week)
        except Exception:
            week_i = None
    # If week is provided but season missing, infer latest season from games/predictions
    if week_i is not None and season_i is None:
        try:
            src = games_df if (games_df is not None and not games_df.empty) else pred_df
            if src is not None and not src.empty and 'season' in src.columns and not src['season'].isna().all():
                season_i = int(src['season'].max())
        except Exception:
            pass
    # Default to the current (season, week) inferred by date when no explicit week/date
    if week_i is None and not date:
        try:
            src = games_df if (games_df is not None and not games_df.empty) else pred_df
            inferred = _infer_current_season_week(src) if (src is not None and not src.empty) else None
            if inferred is not None:
                season_i, week_i = int(inferred[0]), int(inferred[1])
            else:
                # Fallback: latest season, week 1
                if src is not None and not src.empty and 'season' in src.columns and not src['season'].isna().all():
                    if season_i is None:
                        season_i = int(src['season'].max())
                week_i = 1
        except Exception:
            # Last-resort fallback
            week_i = 1

    # Build combined view and enrich with model preds + odds/weather
    view_df = _build_week_view(pred_df, games_df, season_i, week_i)
    view_df = _attach_model_predictions(view_df)
    if view_df is None or view_df.empty:
        return {"rows": 0, "data": []}, 200
    # Optional date filter against combined view
    if date:
        try:
            if "game_date" in view_df.columns:
                view_df = view_df[view_df["game_date"].astype(str).str[:10] == str(date)]
            elif "date" in view_df.columns:
                view_df = view_df[view_df["date"].astype(str).str[:10] == str(date)]
        except Exception:
            pass

    # Global filter overrides
    min_ev = request.args.get("min_ev")
    one = request.args.get("one_per_game")
    if min_ev:
        os.environ['RECS_MIN_EV_PCT'] = str(min_ev)
    if one is not None:
        os.environ['RECS_ONE_PER_GAME'] = str(one)
    # Build recs
    all_recs: List[Dict[str, Any]] = []
    for _, row in view_df.iterrows():
        try:
            recs = _compute_recommendations_for_row(row)
            all_recs.extend(recs)
        except Exception:
            continue
    # If nothing qualifies and no explicit min_ev was provided, relax threshold once
    if not all_recs and not request.args.get("min_ev"):
        try:
            os.environ['RECS_MIN_EV_PCT'] = '1.0'
            for _, row in view_df.iterrows():
                try:
                    recs = _compute_recommendations_for_row(row)
                    all_recs.extend(recs)
                except Exception:
                    continue
        except Exception:
            pass
    # Optional sorting: time | type | odds | ev | level
    sort_param = (request.args.get("sort") or "").lower()
    type_order = {"MONEYLINE": 0, "SPREAD": 1, "TOTAL": 2}
    def sort_key_time(r: Dict[str, Any]):
        try:
            return pd.to_datetime(r.get("game_date"), errors='coerce')
        except Exception:
            return pd.NaT
    def sort_key_type(r: Dict[str, Any]):
        return type_order.get(str(r.get("type")).upper(), 99)
    def sort_key_odds(r: Dict[str, Any]):
        o = r.get("odds")
        try:
            return float(o) if o is not None and not pd.isna(o) else -9999
        except Exception:
            return -9999
    def sort_key_ev(r: Dict[str, Any]):
        v = r.get("ev_pct")
        try:
            return float(v) if v is not None else -9999
        except Exception:
            return -9999
    def sort_key_level(r: Dict[str, Any]):
        return _tier_to_num(r.get("confidence"))
    if sort_param == "time":
        all_recs.sort(key=sort_key_time)
    elif sort_param == "type":
        all_recs.sort(key=sort_key_type)
    elif sort_param == "odds":
        all_recs.sort(key=sort_key_odds, reverse=True)
    elif sort_param == "ev":
        all_recs.sort(key=sort_key_ev, reverse=True)
    elif sort_param == "level":
        all_recs.sort(key=sort_key_level, reverse=True)
    else:
        # Default: by confidence then EV desc
        def sort_key(r: Dict[str, Any]):
            w = r.get("sort_weight") or (0, -999)
            return (w[0], w[1])
        all_recs.sort(key=sort_key, reverse=True)
    for r in all_recs:
        r.pop("sort_weight", None)
    return {"rows": len(all_recs), "data": all_recs}, 200


@app.route("/recommendations")
def recommendations_page():
    """HTML page for recommendations, sorted and grouped by confidence.
    Build from combined games + predictions view to ensure lines/finals are available.
    """
    pred_df = _load_predictions()
    games_df = _load_games()

    # Parse filters
    season = request.args.get("season")
    week = request.args.get("week")
    date = request.args.get("date")
    active_week = None

    # Determine default season/week
    season_i = None
    week_i = None
    if season:
        try:
            season_i = int(season)
        except Exception:
            season_i = None
    if week:
        try:
            week_i = int(week)
        except Exception:
            week_i = None
    # If week provided but season missing, infer latest season
    if week_i is not None and season_i is None:
        try:
            src = games_df if (games_df is not None and not games_df.empty) else pred_df
            if src is not None and not src.empty and 'season' in src.columns and not src['season'].isna().all():
                season_i = int(src['season'].max())
        except Exception:
            season_i = None
    # Default to inferred "current" week when no explicit week/date provided
    if week_i is None and not date:
        try:
            src = games_df if (games_df is not None and not games_df.empty) else pred_df
            inferred = _infer_current_season_week(src) if (src is not None and not src.empty) else None
            if inferred is not None:
                season_i, week_i = int(inferred[0]), int(inferred[1])
            else:
                # Fallback: latest season, week 1
                if src is not None and not src.empty and 'season' in src.columns and not src['season'].isna().all():
                    if season_i is None:
                        season_i = int(src['season'].max())
                week_i = 1
        except Exception:
            # Last-resort fallback
            week_i = 1
    active_week = week_i

    # Build combined view and optionally filter by date
    view_df = _build_week_view(pred_df, games_df, season_i, week_i)
    view_df = _attach_model_predictions(view_df)
    if view_df is None:
        view_df = pd.DataFrame()
    if date and not view_df.empty:
        try:
            if "game_date" in view_df.columns:
                view_df = view_df[view_df["game_date"].astype(str).str[:10] == str(date)]
            elif "date" in view_df.columns:
                view_df = view_df[view_df["date"].astype(str).str[:10] == str(date)]
        except Exception:
            pass

    # Global filter overrides (query)
    min_ev = request.args.get("min_ev")
    one = request.args.get("one_per_game")
    if min_ev:
        os.environ['RECS_MIN_EV_PCT'] = str(min_ev)
    if one is not None:
        os.environ['RECS_ONE_PER_GAME'] = str(one)
    all_recs: List[Dict[str, Any]] = []
    for _, row in (view_df if view_df is not None else pd.DataFrame()).iterrows():
        try:
            recs = _compute_recommendations_for_row(row)
            all_recs.extend(recs)
        except Exception:
            continue
    # If still empty and no explicit min_ev in query, relax threshold once
    if not all_recs and not request.args.get("min_ev"):
        try:
            os.environ['RECS_MIN_EV_PCT'] = '0.5'
            for _, row in (view_df if view_df is not None else pd.DataFrame()).iterrows():
                try:
                    recs = _compute_recommendations_for_row(row)
                    all_recs.extend(recs)
                except Exception:
                    continue
        except Exception:
            pass
    # Sort per query: time | type | odds | ev | level; default confidence->EV
    sort_param = (request.args.get("sort") or "").lower()
    type_order = {"MONEYLINE": 0, "SPREAD": 1, "TOTAL": 2}
    def sort_key_time(r: Dict[str, Any]):
        try:
            return pd.to_datetime(r.get("game_date"), errors='coerce')
        except Exception:
            return pd.NaT
    def sort_key_type(r: Dict[str, Any]):
        return type_order.get(str(r.get("type")).upper(), 99)
    def sort_key_odds(r: Dict[str, Any]):
        o = r.get("odds")
        try:
            return float(o) if o is not None and not pd.isna(o) else -9999
        except Exception:
            return -9999
    def sort_key_ev(r: Dict[str, Any]):
        v = r.get("ev_pct")
        try:
            return float(v) if v is not None else -9999
        except Exception:
            return -9999
    def sort_key_level(r: Dict[str, Any]):
        return _tier_to_num(r.get("confidence"))
    if sort_param == "time":
        all_recs.sort(key=sort_key_time)
    elif sort_param == "type":
        all_recs.sort(key=sort_key_type)
    elif sort_param == "odds":
        all_recs.sort(key=sort_key_odds, reverse=True)
    elif sort_param == "ev":
        all_recs.sort(key=sort_key_ev, reverse=True)
    elif sort_param == "level":
        all_recs.sort(key=sort_key_level, reverse=True)
    else:
        def sort_key(r: Dict[str, Any]):
            w = r.get("sort_weight") or (0, -999)
            return (w[0], w[1])
        all_recs.sort(key=sort_key, reverse=True)
    for r in all_recs:
        r.pop("sort_weight", None)

    groups: Dict[str, List[Dict[str, Any]]] = {"High": [], "Medium": [], "Low": [], "": []}
    for r in all_recs:
        c = r.get("confidence") or ""
        if c not in groups:
            groups[c] = []
        groups[c].append(r)

    # Build accuracy & ROI metrics per tier with weighted stakes
    stake_map = {'High': 100.0, 'Medium': 50.0, 'Low': 25.0}

    def american_profit(stake: float, odds: Any) -> Optional[float]:
        try:
            if odds is None or (isinstance(odds, float) and pd.isna(odds)):
                odds = -110  # fallback
            o = float(odds)
            if o > 0:
                return stake * (o / 100.0)
            else:
                return stake * (100.0 / abs(o))
        except Exception:
            return None

    def tier_metrics(tier: str) -> Dict[str, Any]:
        items = groups.get(tier, [])
        done = [x for x in items if x.get('result') in {'Win','Loss','Push'}]
        wins = sum(1 for x in done if x.get('result') == 'Win')
        losses = sum(1 for x in done if x.get('result') == 'Loss')
        pushes = sum(1 for x in done if x.get('result') == 'Push')
        played = wins + losses  # exclude pushes from accuracy denominator
        acc = (wins / played * 100.0) if played > 0 else None
        stake_total = 0.0
        profit_total = 0.0
        for x in done:
            stake = stake_map.get(tier, 25.0)
            res = x.get('result')
            odds_val = x.get('odds')
            if res == 'Win':
                prof = american_profit(stake, odds_val)
                if prof is None:
                    prof = stake * (100.0/110.0)  # approx -110
                profit_total += prof
                stake_total += stake
            elif res == 'Loss':
                profit_total -= stake
                stake_total += stake
            elif res == 'Push':
                # Stake returned; no profit, but should not count toward stake turnover for ROI denominator? Usually stake at risk counts.
                stake_total += 0.0
        roi_pct = (profit_total / stake_total * 100.0) if stake_total > 0 else None
        return {
            'tier': tier,
            'total': len(items),
            'resolved': len(done),
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'accuracy_pct': acc,
            'roi_pct': roi_pct,
            'stake_total': stake_total,
            'profit_total': profit_total,
        }

    accuracy_summary = {t: tier_metrics(t) for t in ['High','Medium','Low']}
    # Overall metrics (aggregate resolved bets across tiers)
    overall = {'tier': 'Overall','total':0,'resolved':0,'wins':0,'losses':0,'pushes':0,'accuracy_pct':None,'roi_pct':None,'stake_total':0.0,'profit_total':0.0}
    for t in ['High','Medium','Low']:
        m = accuracy_summary.get(t, {})
        for k in ['total','resolved','wins','losses','pushes','stake_total','profit_total']:
            overall[k] += m.get(k, 0) or 0
    played_overall = overall['wins'] + overall['losses']
    if played_overall > 0:
        overall['accuracy_pct'] = overall['wins'] / played_overall * 100.0
    if overall['stake_total'] > 0:
        overall['roi_pct'] = overall['profit_total'] / overall['stake_total'] * 100.0
    accuracy_summary['Overall'] = overall

    return render_template(
        "recommendations.html",
        recs=all_recs,
        groups=groups,
        have_data=len(all_recs) > 0,
        week=active_week,
        sort=sort_param,
        accuracy=accuracy_summary,
    )


def _build_cards(view_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Construct card dictionaries from a weekly view DataFrame.
    This encapsulates the display and reconciliation logic used by the index view
    and exposes it for programmatic or API consumption.
    """
    cards: List[Dict[str, Any]] = []
    assets = _load_team_assets()
    stad_map = _load_stadium_meta_map()
    if view_df is not None and not view_df.empty:
        # Backward compatibility mapping: model earlier produced pred_home_score/pred_away_score
        # but card logic prefers pred_home_points/pred_away_points. Populate points if missing.
        try:
            if 'pred_home_points' not in view_df.columns and 'pred_home_score' in view_df.columns:
                view_df['pred_home_points'] = view_df['pred_home_score']
            if 'pred_away_points' not in view_df.columns and 'pred_away_score' in view_df.columns:
                view_df['pred_away_points'] = view_df['pred_away_score']
        except Exception:
            pass
        for _, r in view_df.iterrows():
            def g(key: str, *alts: str, default=None):
                for k in (key, *alts):
                    if k in view_df.columns:
                        v = r.get(k)
                        # prefer first non-null value
                        if v is not None and not (isinstance(v, float) and pd.isna(v)):
                            return v
                return default

            # Compute assessments
            # Normalize model team scores
            ph = g("pred_home_points", "pred_home_score")
            pa = g("pred_away_points", "pred_away_score")
            margin = None
            winner = None
            total_pred = g("pred_total")
            # Weather-aware tweak for upcoming outdoor games: small downward adjustment for high precip/wind
            try:
                # Only apply to non-final games with a model total
                is_final_now = False
                _st = g("status")
                if _st is not None and str(_st).upper() == 'FINAL':
                    is_final_now = True
                if (g("home_score") is not None and not (isinstance(g("home_score"), float) and pd.isna(g("home_score")))) \
                   and (g("away_score") is not None and not (isinstance(g("away_score"), float) and pd.isna(g("away_score")))):
                    is_final_now = True
                if (total_pred is not None) and (not is_final_now):
                    roof_ctx = g("stadium_roof", "roof")
                    is_dome_like = False
                    if roof_ctx is not None and not (isinstance(roof_ctx, float) and pd.isna(roof_ctx)):
                        srf = str(roof_ctx).strip().lower()
                        is_dome_like = srf in {"dome","indoor","closed","retractable-closed"}
                    if not is_dome_like:
                        precip_ctx = g("wx_precip_pct", "precip_pct")
                        wind_ctx = g("wx_wind_mph", "wind_mph")
                        adj = 0.0
                        try:
                            if precip_ctx is not None and not (isinstance(precip_ctx, float) and pd.isna(precip_ctx)):
                                p = float(precip_ctx)
                                adj += -2.5 * max(0.0, min(p, 100.0)) / 100.0
                        except Exception:
                            pass
                        try:
                            if wind_ctx is not None and not (isinstance(wind_ctx, float) and pd.isna(wind_ctx)):
                                w = float(wind_ctx)
                                over = max(0.0, w - 10.0)
                                adj += -0.10 * over
                        except Exception:
                            pass
                        try:
                            tp = float(total_pred)
                            total_pred = max(0.0, tp + adj)
                        except Exception:
                            pass
            except Exception:
                pass
            if ph is not None and pa is not None and pd.notna(ph) and pd.notna(pa):
                try:
                    margin = float(ph) - float(pa)
                    if margin > 0:
                        winner = g("home_team")
                    elif margin < 0:
                        winner = g("away_team")
                    else:
                        winner = "Tie"
                except Exception:
                    pass

            # Normalize market lines
            # Final games: prefer closing lines. Upcoming: allow close_* as fallback if market/base missing.
            _hs = g("home_score"); _as = g("away_score")
            _is_final = (_hs is not None and not (isinstance(_hs, float) and pd.isna(_hs))) and (_as is not None and not (isinstance(_as, float) and pd.isna(_as)))
            if _is_final:
                m_spread = g("close_spread_home", "market_spread_home", "spread_home", "open_spread_home")
                m_total = g("close_total", "market_total", "total", "open_total")
            else:
                m_spread = g("market_spread_home", "spread_home", "open_spread_home", "close_spread_home")
                if m_spread is None:
                    m_spread = g("close_spread_home")
                m_total = g("market_total", "total", "open_total", "close_total")
                if m_total is None:
                    m_total = g("close_total")
            edge_spread = None
            edge_total = None
            try:
                if margin is not None and m_spread is not None and pd.notna(m_spread):
                    # Positive edge means model likes home side vs market (home covers if margin + spread > 0)
                    edge_spread = float(margin) + float(m_spread)
                if total_pred is not None and m_total is not None and pd.notna(m_total):
                    edge_total = float(total_pred) - float(m_total)
            except Exception:
                pass

            # Picks summary
            pick_spread = None
            pick_total = None
            try:
                if edge_spread is not None:
                    if edge_spread > 0.5:
                        pick_spread = f"{g('home_team')} covers by {edge_spread:+.1f}"
                    elif edge_spread < -0.5:
                        pick_spread = f"{g('away_team')} covers by {(-edge_spread):+.1f}"
                    else:
                        pick_spread = "No ATS edge"
                if edge_total is not None:
                    if edge_total > 0.5:
                        pick_total = f"Over by {edge_total:+.1f}"
                    elif edge_total < -0.5:
                        pick_total = f"Under by {(-edge_total):+.1f}"
                    else:
                        pick_total = "No total edge"
            except Exception:
                pass

            # Quarter/half breakdown (optional)
            quarters: List[Dict[str, Any]] = []
            for i in (1, 2, 3, 4):
                hq = g(f"pred_home_q{i}")
                aq = g(f"pred_away_q{i}")
                tq = g(f"pred_q{i}_total")
                wq = g(f"pred_q{i}_winner")
                if hq is not None or aq is not None or tq is not None:
                    quarters.append({
                        "label": f"Q{i}",
                        "home": hq,
                        "away": aq,
                        "total": tq,
                        "winner": wq,
                    })
            half1 = g("pred_h1_total")
            half2 = g("pred_h2_total")

            home = g("home_team")
            away = g("away_team")
            a_home = assets.get(str(home), {}) if home else {}
            a_away = assets.get(str(away), {}) if away else {}

            def logo_url(asset: Dict[str, Any]) -> Optional[str]:
                # If a custom logo provided, prefer it; else fallback to a generic placeholder
                if asset.get("logo"):
                    return asset.get("logo")
                abbr = asset.get("abbr")
                if abbr:
                    # Placeholder pattern; replace with your CDN if desired
                    espn_map = {"WAS": "wsh"}
                    code = espn_map.get(abbr.upper(), abbr.lower())
                    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{code}.png"
                return None

            # Actuals if present (for past games)
            actual_home = g("home_score")
            actual_away = g("away_score")
            actual_total = None
            actual_margin = None
            if actual_home is not None and actual_away is not None and pd.notna(actual_home) and pd.notna(actual_away):
                try:
                    actual_home_f = float(actual_home)
                    actual_away_f = float(actual_away)
                    actual_total = actual_home_f + actual_away_f
                    actual_margin = actual_home_f - actual_away_f
                except Exception:
                    pass

            # Status text
            status_text = "FINAL" if actual_total is not None else "Scheduled"

            # Winner correctness
            winner_correct = None
            if winner and actual_margin is not None:
                actual_winner = home if actual_margin > 0 else (away if actual_margin < 0 else "Tie")
                winner_correct = (winner == actual_winner)

            # ATS correctness
            ats_text = None
            ats_correct = None
            if m_spread is not None and pd.notna(m_spread):
                # Model side
                model_side = "Home" if (edge_spread is not None and edge_spread >= 0) else ("Away" if edge_spread is not None else None)
                model_team = home if model_side == "Home" else (away if model_side == "Away" else None)
                # Actual cover
                if actual_margin is not None:
                    # Home covers if actual margin + spread > 0
                    cover_val = actual_margin + float(m_spread)
                    actual_side = "Home" if cover_val > 0 else ("Away" if cover_val < 0 else "Push")
                    if model_side:
                        ats_correct = (model_side == actual_side) if actual_side != "Push" else None
                        if actual_side == "Push":
                            ats_text = "ATS: Push"
                        else:
                            actual_team = home if actual_side == "Home" else away
                            ats_text = f"ATS: {actual_team}"
                # Default text if no actuals
                if ats_text is None and model_side is not None:
                    if model_team:
                        ats_text = f"ATS: {model_team} (model)"

            # Totals correctness (O/U)
            ou_text = None
            totals_text = None
            totals_correct = None
            if m_total is not None and pd.notna(m_total):
                model_ou = "Over" if (edge_total is not None and edge_total > 0) else ("Under" if edge_total is not None else None)
                ou_text = f"O/U: {float(m_total):.2f} • Model: {model_ou or '—'} (Edge {edge_total:+.2f})" if edge_total is not None else f"O/U: {float(m_total):.2f}"
                if actual_total is not None:
                    if actual_total > float(m_total):
                        actual_ou = "Over"
                    elif actual_total < float(m_total):
                        actual_ou = "Under"
                    else:
                        actual_ou = "Push"
                    if model_ou and actual_ou != "Push":
                        totals_correct = (model_ou == actual_ou)
                        totals_text = f"Totals: {model_ou}"
                    elif actual_ou == "Push":
                        totals_text = "Totals: Push"

            # Win prob text (market-blended + shrink toward 0.5 for display)
            wp_home = g("pred_home_win_prob", "prob_home_win", default=None)
            try:
                if wp_home is not None and pd.notna(wp_home):
                    wp_home = float(wp_home)
                    # Blend with implied market probability if odds are present
                    ml_home_val = g("moneyline_home")
                    ml_away_val = g("moneyline_away")
                    mkt_ph, _ = _implied_probs_from_moneylines(ml_home_val, ml_away_val)
                    try:
                        beta_wp = float(os.environ.get('WP_MARKET_BLEND', '0.50'))  # 50% market weight by default
                    except Exception:
                        beta_wp = 0.50
                    p_home_eff = ((1.0 - beta_wp) * wp_home + beta_wp * mkt_ph) if mkt_ph is not None else wp_home
                    # Shrink toward 0.5 to avoid overconfident display
                    try:
                        shrink_wp = float(os.environ.get('WP_SHRINK', '0.35'))
                    except Exception:
                        shrink_wp = 0.35
                    p_home_disp = 0.5 + (p_home_eff - 0.5) * (1.0 - shrink_wp)
                    # Constrain within a band around market implied prob, if available
                    if mkt_ph is not None:
                        try:
                            band = float(os.environ.get('WP_MARKET_BAND', '0.12'))
                        except Exception:
                            band = 0.12
                        lower = max(0.0, mkt_ph - band)
                        upper = min(1.0, mkt_ph + band)
                        p_home_disp = min(max(p_home_disp, lower), upper)
                    # Hard clamp to avoid extreme display
                    try:
                        hard_min = float(os.environ.get('WP_HARD_MIN', '0.15'))
                        hard_max = float(os.environ.get('WP_HARD_MAX', '0.85'))
                    except Exception:
                        hard_min, hard_max = 0.15, 0.85
                    p_home_disp = max(hard_min, min(hard_max, p_home_disp))
                    wp_text = f"Win Prob: Away {((1.0 - p_home_disp)*100):.1f}% / Home {(p_home_disp*100):.1f}%"
                else:
                    wp_text = None
            except Exception:
                wp_text = None

            # Venue + datetime (best-effort)
            game_date = g("game_date", "date")
            # Stadium/TZ from meta with per-game override support
            smeta = stad_map.get(str(home), {}) if home else {}
            # load overrides (reload each request to reflect file changes)
            loc_ovr = _load_location_overrides()
            ovr = None
            gid = str(g("game_id")) if g("game_id") is not None else None
            if gid and loc_ovr['by_game_id']:
                ovr = loc_ovr['by_game_id'].get(gid)
            if ovr is None and game_date and home and away:
                # Normalize date to YYYY-MM-DD to match overrides file
                try:
                    date_key = pd.to_datetime(game_date, errors='coerce').date().isoformat()
                except Exception:
                    date_key = str(game_date)
                ovr = loc_ovr['by_match'].get((date_key, str(home), str(away))) if loc_ovr['by_match'] else None

            stadium = (ovr.get('venue') if (ovr and ovr.get('venue')) else (smeta.get('stadium') or home))
            tz = (ovr.get('tz') if (ovr and ovr.get('tz')) else (smeta.get('tz') or g("tz")))  # prefer override, then meta
            date_str = _format_game_datetime(game_date, tz)
            # location suffix if provided (city/country)
            loc_suffix = None
            if ovr:
                city = ovr.get('city')
                country = ovr.get('country')
                if city or country:
                    if city and country:
                        loc_suffix = f"{city}, {country}"
                    else:
                        loc_suffix = city or country
            venue_text = f"Venue: {stadium} • {date_str}" if date_str else f"Venue: {stadium}"
            if loc_suffix:
                venue_text = f"{venue_text} ({loc_suffix})"
            if ovr and ovr.get('neutral_site') not in (None, '', False, 0, '0', 'False', 'false', 'FALSE'):
                venue_text = f"{venue_text} • Neutral site"

            # Weather line
            wt_parts: List[str] = []
            temp_v = g("wx_temp_f", "temp_f", "temperature_f")
            wind_v = g("wx_wind_mph", "wind_mph")
            precip_v = g("wx_precip_pct", "precip_pct")
            if temp_v is not None and pd.notna(temp_v):
                try:
                    wt_parts.append(f"{float(temp_v):.0f}°F")
                except Exception:
                    pass
            if wind_v is not None and pd.notna(wind_v):
                try:
                    wt_parts.append(f"{float(wind_v):.0f} mph wind")
                except Exception:
                    pass
            if precip_v is not None and pd.notna(precip_v):
                try:
                    wt_parts.append(f"Precip {float(precip_v):.0f}%")
                except Exception:
                    pass
            # Placeholder for weather delta if available in future
            weather_text = f"Weather: {' • '.join(wt_parts)}" if wt_parts else None

            # Total diff (model vs actual) if both present
            total_diff = None
            try:
                if total_pred is not None and actual_total is not None:
                    total_diff = abs(float(total_pred) - float(actual_total))
            except Exception:
                total_diff = None

            # Roof/surface override for weather context
            roof_val = (ovr.get('roof') if (ovr and ovr.get('roof')) else g("stadium_roof", "roof"))
            surface_val = (ovr.get('surface') if (ovr and ovr.get('surface')) else g("surface"))

            cards.append({
                "season": g("season"),
                "week": g("week"),
                "game_date": game_date,
                "game_id": g("game_id"),
                "home_team": home,
                "away_team": away,
                "pred_home_points": ph,
                "pred_away_points": pa,
                "pred_total": total_pred,
                "pred_home_win_prob": g("pred_home_win_prob", "prob_home_win", default=None),
                "prediction_source": g("prediction_source"),
                "market_spread_home": m_spread,
                "market_total": m_total,
                "display_spread_home": m_spread,
                "display_total": m_total,
                "pred_margin": margin,
                "pred_winner": winner,
                # Confidence (overall per-game)
                "game_confidence": g("game_confidence"),
                "edge_spread": edge_spread,
                "edge_total": edge_total,
                "pick_spread": pick_spread,
                "pick_total": pick_total,
                # Weather
                "stadium_roof": roof_val,
                "stadium_surface": surface_val,
                "wx_temp_f": g("wx_temp_f", "temp_f", "temperature_f"),
                "wx_wind_mph": g("wx_wind_mph", "wind_mph"),
                "wx_precip_pct": g("wx_precip_pct", "precip_pct"),
                # Colors
                "home_color": a_home.get("primary"),
                "home_color2": a_home.get("secondary"),
                "away_color": a_away.get("primary"),
                "away_color2": a_away.get("secondary"),
                "home_logo": logo_url(a_home),
                "away_logo": logo_url(a_away),
                # Periods
                "quarters": quarters,
                "half1_total": half1,
                "half2_total": half2,
                # Actuals & status
                "home_score": actual_home,
                "away_score": actual_away,
                "actual_total": actual_total,
                "status_text": status_text,
                # Assessments strings
                "wp_text": wp_text,
                "wp_blended": p_home_disp if 'p_home_disp' in locals() else None,
                "winner_correct": winner_correct,
                "ats_text": ats_text,
                "ats_correct": ats_correct,
                "ou_text": ou_text,
                "totals_text": totals_text,
                "totals_correct": totals_correct,
                # Venue text
                "venue_text": venue_text,
                "weather_text": weather_text,
                "total_diff": total_diff,
                # Extended weather (precip type / sky)
                "wx_precip_type": g("wx_precip_type"),
                "wx_sky": g("wx_sky"),
                # Odds (sanitize NaN -> None; cast to int for display)
                "moneyline_home": (int(g("moneyline_home")) if (g("moneyline_home") is not None and not pd.isna(g("moneyline_home"))) else None),
                "moneyline_away": (int(g("moneyline_away")) if (g("moneyline_away") is not None and not pd.isna(g("moneyline_away"))) else None),
                "close_spread_home": g("close_spread_home"),
                "close_total": g("close_total"),
                # Implied probabilities (computed below when possible)
                "implied_home_prob": None,
                "implied_away_prob": None,
                # Recommendations (EV-based)
                # Winner (moneyline): compute EV for Home/Away with available moneylines
                # Spread (win margin): EV at -110 using logistic prob from margin edge
                # Total: EV at -110 using logistic prob from total edge
            })

            # Compute recommendations and attach to last card
            c = cards[-1]
            # Winner EV
            try:
                p_home = float(wp_home) if (wp_home is not None and pd.notna(wp_home)) else None
            except Exception:
                p_home = None
            ml_home = g("moneyline_home")
            ml_away = g("moneyline_away")
            dec_home = _american_to_decimal(ml_home) if ml_home is not None else None
            dec_away = _american_to_decimal(ml_away) if ml_away is not None else None
            ev_home_ml = ev_away_ml = None
            # Implied probabilities for display
            iph, ipa = _implied_probs_from_moneylines(ml_home, ml_away)
            c["implied_home_prob"] = float(iph) if iph is not None else None
            c["implied_away_prob"] = float(ipa) if ipa is not None else None
            p_home_eff = None
            if p_home is not None:
                # RAW model probability (no blending)
                p_home_eff = p_home
            # Derive model winner strictly from predicted point margin when available
            # (earlier we computed 'margin' = pred_home_points - pred_away_points when possible)
            model_winner_by_margin = None
            try:
                if margin is not None and pd.notna(margin):
                    if float(margin) > 0:
                        model_winner_by_margin = home
                    elif float(margin) < 0:
                        model_winner_by_margin = away
                    else:
                        model_winner_by_margin = None  # tie
            except Exception:
                model_winner_by_margin = None

            # If we have both a margin-based winner and a probability, but they disagree, flip orientation
            # of the probability so that the probability perspective matches the point-based winner.
            # This guards against upstream field misalignment.
            if p_home_eff is not None and model_winner_by_margin is not None:
                prob_implies_home = (p_home_eff >= 0.5)
                margin_implies_home = (model_winner_by_margin == home)
                if prob_implies_home != margin_implies_home:
                    # Flip perspective
                    p_home_eff = 1.0 - p_home_eff
                    c['debug_prob_flipped_to_match_margin'] = True
                else:
                    c['debug_prob_flipped_to_match_margin'] = False

            # Compute EV ONLY for the margin-based winner side (if probability present)
            model_winner_prob = None
            model_winner_ev_units = None
            if p_home_eff is not None and model_winner_by_margin is not None:
                if model_winner_by_margin == home:
                    model_winner_prob = p_home_eff
                    if dec_home is not None:
                        model_winner_ev_units = _ev_from_prob_and_decimal(model_winner_prob, dec_home)
                elif model_winner_by_margin == away:
                    # Away win probability = 1 - p_home_eff
                    model_winner_prob = 1.0 - p_home_eff
                    if dec_away is not None:
                        model_winner_ev_units = _ev_from_prob_and_decimal(model_winner_prob, dec_away)

            # For transparency also compute the opposite side EV for debugging (not for recommendation)
            if p_home_eff is not None:
                if dec_home is not None:
                    ev_home_ml = _ev_from_prob_and_decimal(p_home_eff, dec_home)
                if dec_away is not None:
                    ev_away_ml = _ev_from_prob_and_decimal(1.0 - p_home_eff, dec_away)
            # Record debug inputs
            c["debug_p_home_model"] = p_home
            c["debug_p_home_eff"] = p_home_eff
            c["debug_dec_home"] = dec_home
            c["debug_dec_away"] = dec_away
            c["debug_ev_home_ml"] = ev_home_ml
            c["debug_ev_away_ml"] = ev_away_ml
            # Choose winner strictly aligned with model unless explicitly disabled
            winner_side = None
            winner_ev = None
            force_align = os.environ.get('RECS_FORCE_MODEL_WINNER', '1').lower() in {'1','true','yes','y'}
            # Determine recommendation using margin-aligned winner
            if model_winner_by_margin is not None and model_winner_ev_units is not None:
                try:
                    min_ml_ev = float(os.environ.get('RECS_ML_MIN_EV', '0.0'))
                except Exception:
                    min_ml_ev = 0.0
                c["model_winner_side"] = model_winner_by_margin
                c["model_winner_ev_units"] = model_winner_ev_units
                c["model_winner_ev_pct"] = model_winner_ev_units * 100.0
                c["ml_min_ev_units"] = min_ml_ev
                c["ml_min_ev_pct"] = min_ml_ev * 100.0
                if model_winner_ev_units >= min_ml_ev and model_winner_ev_units > 0:
                    winner_side, winner_ev = model_winner_by_margin, model_winner_ev_units
                else:
                    winner_side, winner_ev = None, None
            else:
                # Fallback: if no margin-based winner, revert to probability threshold logic
                model_winner_tmp = None
                try:
                    model_winner_tmp = home if (p_home_eff is not None and p_home_eff >= 0.5) else away if p_home_eff is not None else None
                except Exception:
                    model_winner_tmp = None
                if model_winner_tmp is not None:
                    try:
                        min_ml_ev = float(os.environ.get('RECS_ML_MIN_EV', '0.0'))
                    except Exception:
                        min_ml_ev = 0.0
                    # Use pre-computed ev_home_ml / ev_away_ml
                    mw_ev = ev_home_ml if model_winner_tmp == home else ev_away_ml
                    c["model_winner_side"] = model_winner_tmp
                    c["model_winner_ev_units"] = mw_ev
                    c["model_winner_ev_pct"] = (mw_ev * 100.0) if mw_ev is not None else None
                    c["ml_min_ev_units"] = min_ml_ev
                    c["ml_min_ev_pct"] = min_ml_ev * 100.0
                    if mw_ev is not None and mw_ev >= min_ml_ev and mw_ev > 0:
                        winner_side, winner_ev = model_winner_tmp, mw_ev
                    else:
                        winner_side, winner_ev = None, None
            c["rec_winner_side"] = winner_side
            c["rec_winner_ev"] = winner_ev
            # Confidence for this market should reflect EV only; do not inherit game-level confidence
            c["rec_winner_conf"] = _conf_from_ev(winner_ev) if winner_ev is not None else None
            # Difference flag (may be always False if force_align)
            c["rec_winner_differs"] = False

            # Spread (win margin) EV using actual prices when available (fallback to -110)
            ev_spread_home = ev_spread_away = None
            spread = m_spread
            if margin is not None and spread is not None and pd.notna(spread):
                # edge_pts = predicted margin + spread for home side (home covers if margin + spread > 0)
                try:
                    edge_pts = float(margin) + float(spread)
                    scale_margin = float(os.environ.get('NFL_ATS_SIGMA', '9.0'))
                except Exception:
                    edge_pts, scale_margin = None, 9.0
                if edge_pts is not None:
                    p_home_cover = _cover_prob_from_edge(edge_pts, scale_margin)
                    # Shrink toward 0.5 to reduce overconfidence
                    try:
                        shrink = float(os.environ.get('RECS_PROB_SHRINK', '0.35'))
                    except Exception:
                        shrink = 0.35
                    p_home_cover = 0.5 + (p_home_cover - 0.5) * (1.0 - shrink)
                    sh_price = g("spread_home_price")
                    sa_price = g("spread_away_price")
                    dec_home_sp = _american_to_decimal(sh_price) if sh_price is not None and not pd.isna(sh_price) else (1.0 + 100.0/110.0)
                    dec_away_sp = _american_to_decimal(sa_price) if sa_price is not None and not pd.isna(sa_price) else (1.0 + 100.0/110.0)
                    ev_spread_home = _ev_from_prob_and_decimal(p_home_cover, dec_home_sp)
                    ev_spread_away = _ev_from_prob_and_decimal(1.0 - p_home_cover, dec_away_sp)
            spread_side = None
            spread_ev = None
            if ev_spread_home is not None or ev_spread_away is not None:
                cand = [(home or "Home", ev_spread_home), (away or "Away", ev_spread_away)]
                cand = [(s, e) for s, e in cand if e is not None]
                if cand:
                    s, e = max(cand, key=lambda t: t[1])
                    spread_side, spread_ev = s, e
            c["rec_spread_side"] = spread_side
            c["rec_spread_ev"] = spread_ev
            # Confidence for this market should reflect EV only; do not inherit game-level confidence
            c["rec_spread_conf"] = _conf_from_ev(spread_ev) if spread_ev is not None else None

            # Total EV using actual prices when available (fallback to -110)
            ev_over = ev_under = None
            if total_pred is not None and m_total is not None and pd.notna(m_total):
                try:
                    edge_t = float(total_pred) - float(m_total)
                    scale_total = float(os.environ.get('NFL_TOTAL_SIGMA', '10.0'))
                except Exception:
                    edge_t, scale_total = None, 10.0
                if edge_t is not None:
                    p_over = _cover_prob_from_edge(edge_t, scale_total)
                    try:
                        shrink = float(os.environ.get('RECS_PROB_SHRINK', '0.35'))
                    except Exception:
                        shrink = 0.35
                    p_over = 0.5 + (p_over - 0.5) * (1.0 - shrink)
                    to_price = g("total_over_price")
                    tu_price = g("total_under_price")
                    dec_over = _american_to_decimal(to_price) if to_price is not None and not pd.isna(to_price) else (1.0 + 100.0/110.0)
                    dec_under = _american_to_decimal(tu_price) if tu_price is not None and not pd.isna(tu_price) else (1.0 + 100.0/110.0)
                    ev_over = _ev_from_prob_and_decimal(p_over, dec_over)
                    ev_under = _ev_from_prob_and_decimal(1.0 - p_over, dec_under)
            total_side = None
            total_ev = None
            if ev_over is not None or ev_under is not None:
                cand = [("Over", ev_over), ("Under", ev_under)]
                cand = [(s, e) for s, e in cand if e is not None]
                if cand:
                    s, e = max(cand, key=lambda t: t[1])
                    total_side, total_ev = s, e
            c["rec_total_side"] = total_side
            c["rec_total_ev"] = total_ev
            # Confidence for this market should reflect EV only; do not inherit game-level confidence
            c["rec_total_conf"] = _conf_from_ev(total_ev) if total_ev is not None else None
    return cards


@app.route("/")
def index():
    df = _load_predictions()
    games_df = _load_games()
    # Filters
    season_param: Optional[int] = None
    week_param: Optional[int] = None
    sort_param: str = request.args.get("sort") or "date"
    # Default fast mode on Render or when on-demand predictions are disabled, unless explicitly overridden via ?fast=
    fast_qs = request.args.get("fast")
    on_render = str(os.environ.get("RENDER", "")).strip() != ""
    disable_on_request = str(os.environ.get("DISABLE_ON_REQUEST_PREDICTIONS", "0")).lower() in {"1","true","yes","y"}
    if fast_qs is None:
        fast_mode: bool = (on_render or disable_on_request)
    else:
        fast_mode = (fast_qs.lower() in {"1","true","yes","y"})
    try:
        if request.args.get("season"):
            season_param = int(request.args.get("season"))
        if request.args.get("week"):
            week_param = int(request.args.get("week"))
    except Exception:
        pass

    # Default to the current (season, week) inferred by date when no explicit filters
    if season_param is None and week_param is None:
        try:
            src = games_df if (games_df is not None and not games_df.empty) else df
            inferred = _infer_current_season_week(src) if (src is not None and not src.empty) else None
            if inferred is not None:
                season_param, week_param = int(inferred[0]), int(inferred[1])
            else:
                # Fallback: latest season, week 1
                if src is not None and not src.empty and 'season' in src.columns and not src['season'].isna().all():
                    season_param = int(src['season'].max())
                week_param = 1
        except Exception:
            week_param = 1

    # Build combined view from games + predictions for the target week
    try:
        view_df = _build_week_view(df, games_df, season_param, week_param)
        # Always attach odds/weather enrichment so markets (ML/spread/total) render in fast mode too.
        # Heavy model inference is internally skipped inside _attach_model_predictions on Render/disabled envs.
        try:
            view_df = _attach_model_predictions(view_df)
        except Exception as e:
            _log_once('attach-preds-fail', f'_attach_model_predictions failed: {e}')
        # Derive synthetic predictions from market if model outputs entirely absent (skip in fast mode)
        if not fast_mode:
            try:
                view_df = _derive_predictions_from_market(view_df)
            except Exception as e:
                _log_once('derive-from-market-fail', f'_derive_predictions_from_market failed: {e}')
    except Exception as e:
        _log_once('index-fast-fallback', f'index pipeline failed early: {e}')
        view_df = pd.DataFrame()
    if view_df is None:
        view_df = pd.DataFrame()

    # Build cards via helper
    cards: List[Dict[str, Any]] = _build_cards(view_df)

    # Sanitize numerics to avoid Jinja formatting errors on pandas NA in templates
    def _safe_num(v):
        try:
            if v is None:
                return None
            # pd.NA or NaN handling
            if pd.isna(v):
                return None
            return float(v)
        except Exception:
            return None

    numeric_keys = (
        'market_total', 'market_spread_home', 'moneyline_home', 'moneyline_away',
        'implied_home_prob', 'implied_away_prob', 'edge_spread', 'edge_total',
        'pred_total', 'pred_margin', 'total_diff'
    )
    for c in cards:
        for k in numeric_keys:
            if k in c:
                c[k] = _safe_num(c.get(k))

    # Apply sorting
    def _dt_key(card: Dict[str, Any]):
        try:
            return pd.to_datetime(card.get("game_date"), errors='coerce')
        except Exception:
            return pd.NaT
    if sort_param == "date":
        cards.sort(key=_dt_key)
    elif sort_param == "winner":
        cards.sort(key=lambda c: (c.get("rec_winner_ev") if c.get("rec_winner_ev") is not None else float('-inf')), reverse=True)
    elif sort_param == "ats":
        cards.sort(key=lambda c: (abs(c.get("edge_spread")) if c.get("edge_spread") is not None else float('-inf')), reverse=True)
    elif sort_param == "total":
        cards.sort(key=lambda c: (abs(c.get("edge_total")) if c.get("edge_total") is not None else float('-inf')), reverse=True)

    # --- Season-to-date accuracy summary (Overall, High/Med/Low) ---
    # Build recommendations across all completed weeks up to the active week and
    # compute tiered accuracy/ROI similar to recommendations page.
    accuracy_s2d = None
    try:
        if season_param is not None and week_param is not None:
            # Aggregate recommendations across weeks 1..week_param
            all_recs_s2d: List[Dict[str, Any]] = []
            # Small safety bound on weeks
            max_weeks = int(min(max(week_param, 1), 25))
            for wk in range(1, max_weeks + 1):
                try:
                    vw = _build_week_view(df, games_df, season_param, wk)
                    try:
                        vw = _attach_model_predictions(vw)
                    except Exception:
                        pass
                    if not fast_mode:
                        try:
                            vw = _derive_predictions_from_market(vw)
                        except Exception:
                            pass
                except Exception:
                    vw = None
                if vw is None:
                    continue
                for _, row in vw.iterrows():
                    try:
                        recs = _compute_recommendations_for_row(row)
                        all_recs_s2d.extend(recs)
                    except Exception:
                        continue

            # Group and compute metrics per recommendations_page logic
            groups_s2d: Dict[str, List[Dict[str, Any]]] = {"High": [], "Medium": [], "Low": [], "": []}
            for r in all_recs_s2d:
                c = r.get("confidence") or ""
                if c not in groups_s2d:
                    groups_s2d[c] = []
                groups_s2d[c].append(r)

            stake_map = {"High": 100.0, "Medium": 50.0, "Low": 25.0}
            def american_profit(stake: float, odds: Any) -> Optional[float]:
                try:
                    if odds is None or (isinstance(odds, float) and pd.isna(odds)):
                        odds = -110
                    o = float(odds)
                    if o > 0:
                        return stake * (o / 100.0)
                    else:
                        return stake * (100.0 / abs(o))
                except Exception:
                    return None
            def tier_metrics(tier: str) -> Dict[str, Any]:
                items = groups_s2d.get(tier, [])
                done = [x for x in items if x.get('result') in {'Win','Loss','Push'}]
                wins = sum(1 for x in done if x.get('result') == 'Win')
                losses = sum(1 for x in done if x.get('result') == 'Loss')
                pushes = sum(1 for x in done if x.get('result') == 'Push')
                played = wins + losses
                acc = (wins / played * 100.0) if played > 0 else None
                stake_total = 0.0
                profit_total = 0.0
                for x in done:
                    stake = stake_map.get(tier, 25.0)
                    res = x.get('result')
                    odds_val = x.get('odds')
                    if res == 'Win':
                        prof = american_profit(stake, odds_val)
                        if prof is None:
                            prof = stake * (100.0/110.0)
                        profit_total += prof
                        stake_total += stake
                    elif res == 'Loss':
                        profit_total -= stake
                        stake_total += stake
                    elif res == 'Push':
                        stake_total += 0.0
                roi_pct = (profit_total / stake_total * 100.0) if stake_total > 0 else None
                return {
                    'tier': tier,
                    'total': len(items),
                    'resolved': len(done),
                    'wins': wins,
                    'losses': losses,
                    'pushes': pushes,
                    'accuracy_pct': acc,
                    'roi_pct': roi_pct,
                    'stake_total': stake_total,
                    'profit_total': profit_total,
                }

            accuracy_s2d = {t: tier_metrics(t) for t in ['High','Medium','Low']}
            overall = {'tier': 'Overall','total':0,'resolved':0,'wins':0,'losses':0,'pushes':0,'accuracy_pct':None,'roi_pct':None,'stake_total':0.0,'profit_total':0.0}
            for t in ['High','Medium','Low']:
                m = accuracy_s2d.get(t, {})
                for k in ['total','resolved','wins','losses','pushes','stake_total','profit_total']:
                    overall[k] += m.get(k, 0) or 0
            played_overall = overall['wins'] + overall['losses']
            if played_overall > 0:
                overall['accuracy_pct'] = overall['wins'] / played_overall * 100.0
            if overall['stake_total'] > 0:
                overall['roi_pct'] = overall['profit_total'] / overall['stake_total'] * 100.0
            accuracy_s2d['Overall'] = overall
    except Exception:
        accuracy_s2d = None

    return render_template(
        "index.html",
        have_data=len(cards) > 0,
        cards=cards,
        season=season_param,
        week=week_param,
        sort=sort_param,
        total_rows=len(cards),
        fast_mode=fast_mode,
        accuracy_s2d=accuracy_s2d,
    )


@app.route("/api/health/page")
def api_health_page():
    """Profile the main page build steps to help diagnose 502/timeouts."""
    try:
        t0 = time.time()
        df = _load_predictions(); t1 = time.time()
        games_df = _load_games(); t2 = time.time()
        # Infer defaults
        season_param = week_param = None
        try:
            src = games_df if (games_df is not None and not games_df.empty) else df
            inferred = _infer_current_season_week(src) if (src is not None and not src.empty) else None
            if inferred is not None:
                season_param, week_param = int(inferred[0]), int(inferred[1])
        except Exception:
            pass
        vw = _build_week_view(df, games_df, season_param, week_param); t3 = time.time()
        vwe = _attach_model_predictions(vw.copy() if vw is not None else vw); t4 = time.time()
        vwd = _derive_predictions_from_market(vwe.copy() if vwe is not None else vwe); t5 = time.time()
        cards = _build_cards(vwd if vwd is not None else pd.DataFrame()); t6 = time.time()
        return jsonify({
            'timings_ms': {
                'load_predictions': int((t1-t0)*1000),
                'load_games': int((t2-t1)*1000),
                'build_week_view': int((t3-t2)*1000),
                'attach_predictions': int((t4-t3)*1000),
                'derive_from_market': int((t5-t4)*1000),
                'build_cards': int((t6-t5)*1000),
                'total': int((t6-t0)*1000),
            },
            'counts': {
                'games_rows': 0 if games_df is None else len(games_df),
                'predictions_rows': 0 if df is None else len(df),
                'week_rows': 0 if vw is None else (0 if getattr(vw, 'empty', True) else len(vw)),
                'cards': len(cards)
            },
            'season': season_param,
            'week': week_param,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/cards")
def api_cards():
    """Return reconciled game cards for a given season/week as JSON.
    Query params: season, week, sort (date|winner|ats|total)
    """
    df = _load_predictions()
    games_df = _load_games()
    season_param: Optional[int] = None
    week_param: Optional[int] = None
    sort_param: str = request.args.get("sort") or "date"
    try:
        if request.args.get("season"):
            season_param = int(request.args.get("season"))
        if request.args.get("week"):
            week_param = int(request.args.get("week"))
    except Exception:
        pass
    # Default inference if not provided
    if season_param is None or week_param is None:
        try:
            src = games_df if (games_df is not None and not games_df.empty) else df
            inferred = _infer_current_season_week(src) if (src is not None and not src.empty) else None
            if inferred is not None:
                if season_param is None:
                    season_param = int(inferred[0])
                if week_param is None:
                    week_param = int(inferred[1])
        except Exception:
            pass
    view_df = _build_week_view(df, games_df, season_param, week_param)
    view_df = _attach_model_predictions(view_df)
    view_df = _derive_predictions_from_market(view_df)
    if view_df is None:
        view_df = pd.DataFrame()
    cards: List[Dict[str, Any]] = _build_cards(view_df)
    # Sorting
    def _dt_key(card: Dict[str, Any]):
        try:
            return pd.to_datetime(card.get("game_date"), errors='coerce')
        except Exception:
            return pd.NaT
    if sort_param == "date":
        cards.sort(key=_dt_key)
    elif sort_param == "winner":
        cards.sort(key=lambda c: (c.get("rec_winner_ev") if c.get("rec_winner_ev") is not None else float('-inf')), reverse=True)
    elif sort_param == "ats":
        cards.sort(key=lambda c: (abs(c.get("edge_spread")) if c.get("edge_spread") is not None else float('-inf')), reverse=True)
    elif sort_param == "total":
        cards.sort(key=lambda c: (abs(c.get("edge_total")) if c.get("edge_total") is not None else float('-inf')), reverse=True)
    return jsonify({
        "season": season_param,
        "week": week_param,
        "total_rows": len(cards),
        "cards": cards,
    })


@app.route("/table")
def table_view():
    df = _load_predictions()
    season_param = request.args.get("season")
    week_param = request.args.get("week")
    try:
        season_i = int(season_param) if season_param else None
        week_i = int(week_param) if week_param else None
    except Exception:
        season_i, week_i = None, None

    view_df = df.copy()
    if not view_df.empty:
        # Default to current week if no filters given
        if season_i is None and week_i is None:
            inferred = _infer_current_season_week(view_df)
            if inferred is not None:
                season_i, week_i = inferred
        if season_i is not None and "season" in view_df.columns:
            view_df = view_df[view_df["season"] == season_i]
        if week_i is not None and "week" in view_df.columns:
            view_df = view_df[view_df["week"] == week_i]

    show_cols = [
        c for c in [
            "season", "week", "game_date", "away_team", "home_team",
            "pred_away_points", "pred_home_points", "pred_total", "pred_home_win_prob",
            "market_spread_home", "market_total", "game_confidence",
        ] if c in view_df.columns
    ]
    rows = view_df[show_cols].to_dict(orient="records") if not view_df.empty else []

    return render_template(
        "table.html",
        have_data=not view_df.empty,
        total_rows=len(rows),
        rows=rows,
        show_cols=show_cols,
    season=season_i,
    week=week_i,
    )


@app.route("/api/refresh-data", methods=["POST", "GET"])
def refresh_data():
    """Synchronous refresh of predictions by invoking the pipeline.

    Query params:
      train=true  -> also run training before predicting
    """
    train = request.args.get("train", "false").lower() == "true"
    # On minimal web deploys (Render), heavy training libs may be absent.
    if os.environ.get("RENDER", "").lower() in {"1", "true", "yes"}:
        return {"status": "skipped", "reason": "Refresh disabled on Render minimal deploy. Run locally or add full requirements."}, 200
    py = sys.executable or "python"
    env = os.environ.copy()
    # Ensure repo root is on module path
    env["PYTHONPATH"] = str(BASE_DIR)
    cmds = []
    if train:
        cmds.append([py, "-m", "nfl_compare.src.train"])
    cmds.append([py, "-m", "nfl_compare.src.predict"])

    details = []
    rc_total = 0
    for c in cmds:
        try:
            res = subprocess.run(c, cwd=str(BASE_DIR), env=env, capture_output=True, text=True, timeout=600)
            details.append({
                "cmd": " ".join(c),
                "returncode": res.returncode,
                "stdout_tail": res.stdout[-1000:],
                "stderr_tail": res.stderr[-1000:],
            })
            rc_total += res.returncode
            if res.returncode != 0:
                break
        except Exception as e:
            return {"status": "error", "error": str(e), "details": details}, 500

    ok = (rc_total == 0)
    return {"status": "ok" if ok else "error", "details": details, "predictions_path": str(PRED_FILE)}, (200 if ok else 500)


@app.route("/api/refresh-odds", methods=["POST", "GET"])
def refresh_odds():
    """Fetch fresh NFL odds (moneyline/spreads/totals) and re-run predictions.

    Requires ODDS_API_KEY in environment. This will write data/real_betting_lines_YYYY_MM_DD.json
    and then execute the prediction pipeline so UI reflects updated lines.
    """
    # On minimal web deploys (Render), odds/client deps may be absent.
    if os.environ.get("RENDER", "").lower() in {"1", "true", "yes"}:
        return {"status": "skipped", "reason": "Odds refresh disabled on Render minimal deploy. Run locally or enable full requirements."}, 200
    py = sys.executable or "python"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)
    details = []
    cmds = [
        [py, "-m", "nfl_compare.src.odds_api_client"],
        [py, "-m", "nfl_compare.src.predict"],
    ]
    rc_total = 0
    for c in cmds:
        try:
            res = subprocess.run(c, cwd=str(BASE_DIR / "nfl_compare"), env=env, capture_output=True, text=True, timeout=600)
            details.append({
                "cmd": " ".join(c),
                "returncode": res.returncode,
                "stdout_tail": res.stdout[-1000:],
                "stderr_tail": res.stderr[-1000:],
            })
            rc_total += res.returncode
            if res.returncode != 0:
                break
        except Exception as e:
            return {"status": "error", "error": str(e), "details": details}, 500
    ok = (rc_total == 0)
    return {"status": "ok" if ok else "error", "details": details}, (200 if ok else 500)


@app.route("/api/odds-coverage")
def odds_coverage():
    """Report coverage of odds fields across current lines and predictions."""
    from nfl_compare.src.data_sources import load_lines
    try:
        df = load_lines()
    except Exception:
        df = pd.DataFrame()
    total = int(len(df))
    def _cnt(col):
        return int((df[col].notna()).sum()) if (col in df.columns and not df.empty) else 0
    out = {
        "rows": total,
        "moneyline_home": _cnt("moneyline_home"),
        "moneyline_away": _cnt("moneyline_away"),
        "spread_home": _cnt("spread_home"),
    "spread_home_price": _cnt("spread_home_price"),
    "spread_away_price": _cnt("spread_away_price"),
    "total": _cnt("total"),
    "total_over_price": _cnt("total_over_price"),
    "total_under_price": _cnt("total_under_price"),
    }
    return jsonify(out)


@app.route("/api/eval")
def api_eval():
    """Return walk-forward evaluation metrics.

    Behavior:
    - If running on Render (RENDER env true), try to read a cached JSON at nfl_compare/data/eval_summary.json.
      If not present, return a 'skipped' status for safety.
    - Otherwise, execute the evaluator in-process and optionally write/update the cache when write_cache=true.
    Query params:
      - min_weeks (int): minimum weeks of prior data to train per season (default env NFL_EVAL_MIN_WEEKS_TRAIN=4)
      - write_cache (bool): if true, write results to cache file.
    """
    cache_path = DATA_DIR / "eval_summary.json"
    is_render = str(os.environ.get("RENDER", "")).lower() in {"1", "true", "yes"}
    min_weeks = os.environ.get("NFL_EVAL_MIN_WEEKS_TRAIN", "4")
    try:
        if request.args.get("min_weeks"):
            min_weeks = str(int(request.args.get("min_weeks")))
    except Exception:
        pass
    write_cache = str(request.args.get("write_cache", "false")).lower() in {"1","true","yes","y"}

    if is_render:
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                return jsonify({"status": "ok", "from_cache": True, "cache_path": str(cache_path), "data": data})
            except Exception as e:
                return jsonify({"status": "error", "error": str(e)}), 500
        return jsonify({"status": "skipped", "reason": "Evaluation disabled on Render; no cache present.", "cache_path": str(cache_path)}), 200

    # Local/full env: run the evaluation in-process
    try:
        from nfl_compare.scripts.evaluate_walkforward import walkforward_eval
        res = walkforward_eval(min_weeks_train=int(min_weeks))
        if write_cache:
            try:
                cache_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
            except Exception:
                # Non-fatal
                pass
        return jsonify({"status": "ok", "from_cache": False, "data": res})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/venue-info")
def venue_info():
    gid = request.args.get('game_id')
    df = _load_predictions()
    if df is None or df.empty:
        return jsonify({"error": "no data"}), 404
    if gid:
        df = df[df['game_id'] == gid]
    if df.empty:
        return jsonify({"error": "game not found"}), 404
    stad_map = _load_stadium_meta_map()
    loc_ovr = _load_location_overrides()
    out = []
    for _, row in df.iterrows():
        home = str(row.get('home_team'))
        away = str(row.get('away_team'))
        game_date = row.get('game_date') if 'game_date' in df.columns else row.get('date')
        smeta = stad_map.get(home, {}) if home else {}
        # override lookup
        ovr = None
        gid_v = str(row.get('game_id')) if row.get('game_id') is not None else None
        if gid_v and loc_ovr['by_game_id']:
            ovr = loc_ovr['by_game_id'].get(gid_v)
        if ovr is None and game_date and home and away:
            try:
                date_key = pd.to_datetime(game_date, errors='coerce').date().isoformat()
            except Exception:
                date_key = str(game_date)
            ovr = loc_ovr['by_match'].get((date_key, home, away)) if loc_ovr['by_match'] else None
        stadium = (ovr.get('venue') if (ovr and ovr.get('venue')) else (smeta.get('stadium') or home))
        tz = (ovr.get('tz') if (ovr and ovr.get('tz')) else (smeta.get('tz') or row.get('tz')))
        date_str = None
        if game_date is not None:
            try:
                date_str = _format_game_datetime(game_date, tz)
            except Exception:
                date_str = None
        venue_text = f"Venue: {stadium} • {date_str}" if date_str else f"Venue: {stadium}"
        city = ovr.get('city') if ovr else None
        country = ovr.get('country') if ovr else None
        if city or country:
            suffix = f"{city}, {country}" if city and country else (city or country)
            venue_text = f"{venue_text} ({suffix})"
        if ovr and ovr.get('neutral_site') not in (None, '', False, 0, '0', 'False', 'false', 'FALSE'):
            venue_text = f"{venue_text} • Neutral site"
        out.append({
            'game_id': row.get('game_id'),
            'date': str(game_date),
            'home_team': home,
            'away_team': away,
            'stadium': stadium,
            'tz': tz,
            'venue_text': venue_text,
            'override_found': bool(ovr),
            'override': ovr or {},
        })
    return jsonify({'data': out})


@app.route("/api/backfill-close-lines", methods=["POST", "GET"])
def api_backfill_close_lines():
    """Run the backfill script to populate close_spread_home/close_total where missing.

    On Render, this is skipped to avoid heavy operations; run locally instead.
    Returns a brief report including counts updated and the output file path.
    """
    if os.environ.get("RENDER", "").lower() in {"1", "true", "yes"}:
        return jsonify({"status": "skipped", "reason": "Backfill disabled on Render minimal deploy. Run locally."}), 200
    py = sys.executable or "python"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)
    cmd = [py, "-m", "nfl_compare.scripts.backfill_close_lines"]
    try:
        res = subprocess.run(cmd, cwd=str(BASE_DIR), env=env, capture_output=True, text=True, timeout=600)
        stdout_tail = res.stdout[-2000:]
        stderr_tail = res.stderr[-2000:]
        ok = (res.returncode == 0)
        return jsonify({
            "status": "ok" if ok else "error",
            "returncode": res.returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }), (200 if ok else 500)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/debug-week-view")
def api_debug_week_view():
    """Debug endpoint: show per-game recommendation counts for a given season/week/date.
    Query: season, week, date, min_ev
    """
    try:
        pred_df = _load_predictions()
        games_df = _load_games()
        season = request.args.get("season")
        week = request.args.get("week")
        date = request.args.get("date")
        season_i = int(season) if season else None
        week_i = int(week) if week else None
        view_df = _build_week_view(pred_df, games_df, season_i, week_i)
        view_df = _attach_model_predictions(view_df)
        if date and not view_df.empty:
            if "game_date" in view_df.columns:
                view_df = view_df[view_df["game_date"].astype(str).str[:10] == str(date)]
            elif "date" in view_df.columns:
                view_df = view_df[view_df["date"].astype(str).str[:10] == str(date)]
        # Optional min_ev override
        if request.args.get("min_ev"):
            os.environ['RECS_MIN_EV_PCT'] = str(request.args.get("min_ev"))
        out = []
        for _, row in view_df.iterrows():
            try:
                recs = _compute_recommendations_for_row(row)
            except Exception:
                recs = []
            out.append({
                'game_id': row.get('game_id'),
                'date': row.get('game_date') or row.get('date'),
                'home_team': row.get('home_team'),
                'away_team': row.get('away_team'),
                'recs': len(recs),
                'has_ml': bool(pd.notna(row.get('moneyline_home')) or pd.notna(row.get('moneyline_away'))),
                'has_spread': bool(pd.notna(row.get('close_spread_home')) or pd.notna(row.get('market_spread_home')) or pd.notna(row.get('spread_home'))),
                'has_total': bool(pd.notna(row.get('close_total')) or pd.notna(row.get('market_total')) or pd.notna(row.get('total'))),
                'pred_total': row.get('pred_total'),
                'pred_home_win_prob': row.get('pred_home_win_prob') or row.get('prob_home_win'),
            })
        return jsonify({'rows': len(out), 'data': out})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/inspect-game")
def api_inspect_game():
    """Inspect a single game's merged fields to debug odds missing issues.
    Query: game_id or (season, week, home_team, away_team)
    """
    try:
        pred_df = _load_predictions()
        games_df = _load_games()
        gid = request.args.get('game_id')
        season = request.args.get('season')
        week = request.args.get('week')
        home = request.args.get('home_team')
        away = request.args.get('away_team')
        season_i = int(season) if season else None
        week_i = int(week) if week else None
        view_df = _build_week_view(pred_df, games_df, season_i, week_i)
        view_df = _attach_model_predictions(view_df)
        if view_df is None or view_df.empty:
            return jsonify({'rows': 0, 'data': []})
        df = view_df.copy()
        if gid and 'game_id' in df.columns:
            df = df[df['game_id'].astype(str) == str(gid)]
        if not gid and home and away and {'home_team','away_team'}.issubset(df.columns):
            df = df[(df['home_team'].astype(str) == str(home)) & (df['away_team'].astype(str) == str(away))]
        # Return a compact set of fields
        keep = [c for c in [
            'game_id','season','week','game_date','date','home_team','away_team',
            'moneyline_home','moneyline_away','spread_home','total','spread_home_price','spread_away_price','total_over_price','total_under_price',
            'close_spread_home','close_total',
            'home_score','away_score'
        ] if c in df.columns]
        data = df[keep].to_dict(orient='records') if keep else df.to_dict(orient='records')

        # Add candidate match info from lines.csv to help debug mismatches (best-effort)
        debug = {}
        try:
            import pandas as _pd
            csv_fp = BASE_DIR / 'nfl_compare' / 'data' / 'lines.csv'
            if csv_fp.exists():
                lines_df = _pd.read_csv(csv_fp)
                # normalize teams
                try:
                    from nfl_compare.src.team_normalizer import normalize_team_name as _norm_team
                    if 'home_team' in lines_df.columns:
                        lines_df['home_team'] = lines_df['home_team'].astype(str).apply(_norm_team)
                    if 'away_team' in lines_df.columns:
                        lines_df['away_team'] = lines_df['away_team'].astype(str).apply(_norm_team)
                except Exception:
                    pass
                # Cast types
                for _c in ('season','week'):
                    if _c in lines_df.columns:
                        lines_df[_c] = _pd.to_numeric(lines_df[_c], errors='coerce').astype('Int64')
                if 'game_id' in lines_df.columns:
                    lines_df['game_id'] = lines_df['game_id'].astype(str)
                if data:
                    rec = data[0]
                    gid = str(rec.get('game_id')) if rec.get('game_id') is not None else None
                    season_v = rec.get('season')
                    week_v = rec.get('week')
                    home_v = rec.get('home_team')
                    away_v = rec.get('away_team')
                    cand = {}
                    if gid and 'game_id' in lines_df.columns:
                        cand['by_game_id'] = lines_df[lines_df['game_id'] == gid].head(1).to_dict(orient='records')
                    mask = None
                    if all(k in lines_df.columns for k in ('season','week','home_team','away_team')):
                        mask = (
                            (lines_df['season'] == season_v) &
                            (lines_df['week'] == week_v) &
                            (lines_df['home_team'] == home_v) &
                            (lines_df['away_team'] == away_v)
                        )
                        cand['by_sw_teams'] = lines_df[mask].head(1).to_dict(orient='records')
                        mask_sw = (
                            (lines_df['season'] == season_v) &
                            (lines_df['week'] == week_v) &
                            (lines_df['home_team'] == away_v) &
                            (lines_df['away_team'] == home_v)
                        )
                        cand['by_swapped_teams'] = lines_df[mask_sw].head(1).to_dict(orient='records')
                    cand['keys'] = {'gid': gid, 'season': season_v, 'week': week_v, 'home': home_v, 'away': away_v}
                    debug['lines_candidates'] = cand
        except Exception:
            pass

        return jsonify({'rows': len(data), 'data': data, 'debug': debug})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/props")
def props_page():
    return render_template("player_props.html")


@app.route('/api/admin/refresh-props', methods=['POST','GET'])
def api_admin_refresh_props():
    """Admin: Fetch Bovada player props and recompute edges for a season/week.

    Query params:
      - season (int, optional): defaults to inferred current season
      - week (int, optional): defaults to inferred current week
    Returns a JSON report with file paths and row counts. Requires ADMIN_KEY/ADMIN_TOKEN.
    """
    if not _admin_auth_ok(request):
        return jsonify({'status': 'forbidden'}), 403
    # On minimal web deploys (Render), skip heavy network/deps for safety
    if os.environ.get("RENDER", "").lower() in {"1","true","yes"}:
        return jsonify({"status": "skipped", "reason": "Disabled on Render minimal deploy."}), 200

    try:
        season_q = request.args.get("season")
        week_q = request.args.get("week")
        season_i = int(season_q) if season_q else None
        week_i = int(week_q) if week_q else None
    except Exception:
        season_i, week_i = None, None

    pred_df = _load_predictions()
    games_df = _load_games()
    if season_i is None or week_i is None:
        try:
            src = games_df if (games_df is not None and not games_df.empty) else pred_df
            inferred = _infer_current_season_week(src) if (src is not None and not src.empty) else None
            if inferred is not None:
                if season_i is None:
                    season_i = int(inferred[0])
                if week_i is None:
                    week_i = int(inferred[1])
        except Exception:
            pass
    if season_i is None or week_i is None:
        return jsonify({"status": "error", "error": "unable to infer season/week"}), 400

    # Paths
    bov_csv = DATA_DIR / f"bovada_player_props_{season_i}_wk{week_i}.csv"
    edges_csv = DATA_DIR / f"edges_player_props_{season_i}_wk{week_i}.csv"

    # 1) Fetch Bovada props via script
    py = sys.executable or "python"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)
    fetch_cmd = [py, str(BASE_DIR / 'scripts' / 'fetch_bovada_props.py'), '--season', str(season_i), '--week', str(week_i), '--out', str(bov_csv)]
    join_cmd = [py, str(BASE_DIR / 'scripts' / 'props_edges_join.py'), '--season', str(season_i), '--week', str(week_i), '--bovada', str(bov_csv), '--out', str(edges_csv), '--data-dir', str(DATA_DIR)]
    details = []
    for cmd in (fetch_cmd, join_cmd):
        try:
            res = subprocess.run(cmd, cwd=str(BASE_DIR), env=env, capture_output=True, text=True, timeout=600)
            details.append({
                'cmd': ' '.join(cmd),
                'returncode': res.returncode,
                'stdout_tail': res.stdout[-1000:],
                'stderr_tail': res.stderr[-1000:],
            })
            if res.returncode != 0:
                return jsonify({'status': 'error', 'step': 'fetch' if cmd is fetch_cmd else 'join', 'details': details}), 500
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500

    # Summarize outputs
    out = {'status': 'ok', 'season': season_i, 'week': week_i, 'bovada_csv': str(bov_csv), 'edges_csv': str(edges_csv), 'details': details}
    try:
        import pandas as _pd
        if bov_csv.exists():
            out['bovada_rows'] = int(len(_pd.read_csv(bov_csv)))
        if edges_csv.exists():
            out['edges_rows'] = int(len(_pd.read_csv(edges_csv)))
    except Exception:
        pass
    return jsonify(out)


@app.route("/api/props/recommendations")
def api_props_recommendations():
    """Return player props recommendations built from precomputed edges CSV.

    Query params:
      - season (int, optional): Defaults to inferred current season.
      - week (int, optional): Defaults to inferred current week.
      - event (str, optional): Filter to a single Bovada event description.
      - home_team/away_team (str, optional): Alternative filter to match a game.
    Response:
      {
        season, week,
        games: [ { event, home_team, away_team }... ],
        rows: N,
        data: [
          {
            player, position, team, opponent, home_team, away_team, event,
            projections: {pass_yards, rush_yards, rec_yards, receptions, any_td_prob},
            plays: [ { market, line, proj, edge, over_price, under_price, side, ev_pct } ... ]
          }, ...
        ]
      }
    """
    try:
        season_q = request.args.get("season")
        week_q = request.args.get("week")
        season_i = int(season_q) if season_q else None
        week_i = int(week_q) if week_q else None
    except Exception:
        season_i, week_i = None, None
    pred_df = _load_predictions()
    games_df = _load_games()
    # Infer defaults if missing
    if season_i is None or week_i is None:
        try:
            src = games_df if (games_df is not None and not games_df.empty) else pred_df
            inferred = _infer_current_season_week(src) if (src is not None and not src.empty) else None
            if inferred is not None:
                if season_i is None:
                    season_i = int(inferred[0])
                if week_i is None:
                    week_i = int(inferred[1])
            else:
                # Fallback: latest season, week 1
                if season_i is None and src is not None and not src.empty and 'season' in src.columns and not src['season'].isna().all():
                    season_i = int(src['season'].max())
                if week_i is None:
                    week_i = 1
        except Exception:
            if week_i is None:
                week_i = 1

    # Locate expected CSVs
    edges_fp = DATA_DIR / f"edges_player_props_{season_i}_wk{week_i}.csv"
    bovada_fp = DATA_DIR / f"bovada_player_props_{season_i}_wk{week_i}.csv"
    preds_fp = DATA_DIR / f"player_props_{season_i}_wk{week_i}.csv"
    # Fallback for older file naming (props_edges_{season}_wk{week}.csv) if primary not present
    edges_fp_used = edges_fp
    try:
        if not edges_fp.exists():
            alt_fp = DATA_DIR / f"props_edges_{season_i}_wk{week_i}.csv"
            if alt_fp.exists():
                edges_fp_used = alt_fp
    except Exception:
        pass

    # Load with graceful fallbacks
    try:
        edges_df = pd.read_csv(edges_fp_used) if edges_fp_used.exists() else pd.DataFrame()
    except Exception:
        edges_df = pd.DataFrame()
    # Flexible CSV reader to handle encoding variants (utf-8, utf-16, latin-1)
    def _read_csv_flex(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except Exception as e:
            # Try common encodings
            encs = [
                ('utf-8', None),
                ('utf-8-sig', None),
                ('utf-16', None),
                ('utf-16le', None),
                ('utf-16be', None),
                ('latin-1', None),
            ]
            for enc, sep in encs:
                try:
                    return pd.read_csv(path, encoding=enc) if sep is None else pd.read_csv(path, encoding=enc, sep=sep)
                except Exception:
                    continue
        return pd.DataFrame()
    try:
        bov_df = _read_csv_flex(bovada_fp) if bovada_fp.exists() else pd.DataFrame()
    except Exception:
        bov_df = pd.DataFrame()
    try:
        preds_df = pd.read_csv(preds_fp) if preds_fp.exists() else pd.DataFrame()
    except Exception:
        preds_df = pd.DataFrame()

    # Attempt to load ladder options (explicit or synthesized) and merge into the working frame
    # so they appear alongside baseline props. Degrade gracefully if file is absent.
    ladder_fp = DATA_DIR / f"ladder_options_{season_i}_wk{week_i}.csv"
    lad_df = pd.DataFrame()
    try:
        if ladder_fp.exists():
            lad_df = pd.read_csv(ladder_fp)
            # Ensure minimal columns exist
            if not lad_df.empty:
                if "is_ladder" not in lad_df.columns:
                    lad_df["is_ladder"] = True
                # Normalize market_key for downstream side selection
                if "market_key" not in lad_df.columns and "market" in lad_df.columns:
                    mk_map = {
                        "receiving yards": "rec_yards",
                        "receptions": "receptions",
                        "rushing yards": "rush_yards",
                        "passing yards": "pass_yards",
                        "anytime td": "any_td",
                        "any time td": "any_td",
                    }
                    try:
                        lad_df["market_key"] = lad_df["market"].astype(str).str.strip().str.lower().map(mk_map).fillna(lad_df["market"].astype(str).str.strip().str.lower())
                    except Exception:
                        pass
                # Coerce numeric line/proj/edge for safety
                for c in ("line","proj","edge"):
                    if c in lad_df.columns:
                        lad_df[c] = pd.to_numeric(lad_df[c], errors="coerce")
    except Exception:
        lad_df = pd.DataFrame()

    # If baseline edges are missing but ladders exist, use ladders as the data source
    try:
        if (edges_df is None or edges_df.empty) and (lad_df is not None and not lad_df.empty):
            edges_df = lad_df.copy()
        elif lad_df is not None and not lad_df.empty:
            # Append ladders to edges for richer play lists
            try:
                edges_df = pd.concat([edges_df, lad_df], ignore_index=True)
            except Exception:
                # Fallback: if concat fails, skip appending ladders
                pass
    except Exception:
        pass

    # Early exit if no edges and no ladders
    if edges_df is None or edges_df.empty:
        return jsonify({
            "season": season_i,
            "week": week_i,
            "games": [],
            "rows": 0,
            "data": [],
            "note": f"edges/ladder data not found or empty: {edges_fp_used} / {ladder_fp}"
        })

    # Normalize helpers
    def _norm(s):
        try:
            return str(s).strip().lower()
        except Exception:
            return None

    # Prefer event/home/away directly from edges CSV; if missing, attach from Bovada
    try:
        need_cols = not {"event","home_team","away_team"}.issubset(set(edges_df.columns))
        if need_cols and not bov_df.empty:
            join_cols = [c for c in ["player", "team", "market", "line"] if c in edges_df.columns and c in bov_df.columns]
            if join_cols:
                edges_df = edges_df.merge(
                    bov_df[[c for c in [*join_cols, "event", "home_team", "away_team", "game_time", "book"] if c in bov_df.columns]],
                    on=join_cols,
                    how="left",
                )
    except Exception:
        pass

    # Compute a basic recommended side per row
    def _rec_side(row):
        mk = str(row.get("market_key") or row.get("market") or "").strip().lower()
        proj = row.get("proj")
        line = row.get("line")
        if mk in {"rec_yards","rush_yards","pass_yards","receptions","pass_attempts","rush_attempts","receiving yards","rushing yards","passing yards","receptions"}:
            try:
                if pd.notna(proj) and pd.notna(line):
                    return "Over" if float(proj) > float(line) else ("Under" if float(proj) < float(line) else None)
            except Exception:
                return None
            return None
        # Markets with EV from Poisson model (interceptions, pass_tds, multi_tds)
        if mk in {"interceptions","pass_tds","multi_tds"}:
            try:
                ov = row.get("over_ev")
                un = row.get("under_ev")
                if (ov is not None and not pd.isna(ov)) or (un is not None and not pd.isna(un)):
                    if (ov is not None and not pd.isna(ov)) and (un is not None and not pd.isna(un)):
                        return "Over" if float(ov) >= float(un) else "Under"
                    return "Over" if (ov is not None and not pd.isna(ov)) else ("Under" if (un is not None and not pd.isna(un)) else None)
            except Exception:
                return None
            # Fallback to proj vs line if EV not available
            try:
                if pd.notna(proj) and pd.notna(line):
                    return "Over" if float(proj) > float(line) else ("Under" if float(proj) < float(line) else None)
            except Exception:
                return None
            return None
        if mk in {"any_td","anytime td","any time td"}:
            try:
                ov = row.get("over_ev")
                un = row.get("under_ev")
                if (ov is not None and not pd.isna(ov)) or (un is not None and not pd.isna(un)):
                    if (ov is not None and not pd.isna(ov)) and (un is not None and not pd.isna(un)):
                        return "Over" if float(ov) >= float(un) else "Under"
                    return "Over" if (ov is not None and not pd.isna(ov)) else ("Under" if (un is not None and not pd.isna(un)) else None)
            except Exception:
                return None
            return None
        return None

    edges_df = edges_df.copy()
    if "market_key" not in edges_df.columns and "market" in edges_df.columns:
        # Minimal mapping to internal keys to aid client display
        mk_map = {
            "receiving yards": "rec_yards",
            "receptions": "receptions",
            "rushing yards": "rush_yards",
            "passing yards": "pass_yards",
            "passing tds": "pass_tds",
            "pass tds": "pass_tds",
            "pass touchdowns": "pass_tds",
            "passing attempts": "pass_attempts",
            "pass attempts": "pass_attempts",
            "rushing attempts": "rush_attempts",
            "rush attempts": "rush_attempts",
            "interceptions": "interceptions",
            "interceptions thrown": "interceptions",
            "rush+rec yards": "rush_rec_yards",
            "rushing + receiving yards": "rush_rec_yards",
            "rush + rec yards": "rush_rec_yards",
            "pass+rush yards": "pass_rush_yards",
            "pass + rush yards": "pass_rush_yards",
            "passing + rushing yards": "pass_rush_yards",
            "targets": "targets",
            "2+ touchdowns": "multi_tds",
            "anytime td": "any_td",
            "any time td": "any_td",
        }
        try:
            edges_df["market_key"] = edges_df["market"].astype(str).str.strip().str.lower().map(mk_map).fillna(edges_df["market"].astype(str).str.strip().str.lower())
        except Exception:
            pass
    # Hide Anytime TD entries with 0% projection before grouping (per requirement)
    try:
        if "market_key" in edges_df.columns and "proj" in edges_df.columns:
            # Ensure numeric comparison is safe
            proj_num = pd.to_numeric(edges_df["proj"], errors="coerce")
            mask_zero_atd = (edges_df["market_key"].astype(str).str.lower() == "any_td") & (proj_num.fillna(-1.0) == 0.0)
            if mask_zero_atd.any():
                edges_df = edges_df.loc[~mask_zero_atd].copy()
    except Exception:
        pass
    try:
        edges_df["rec_side"] = edges_df.apply(_rec_side, axis=1)
    except Exception:
        edges_df["rec_side"] = None

    # Build games list for dropdown (canonicalize labels to avoid duplicates) and include all weekly games
    games = []
    try:
        # Team normalization helpers
        try:
            from nfl_compare.src.team_normalizer import normalize_team_name as _norm_team
        except Exception:
            def _norm_team(x):
                return str(x or "").strip()
        # Build abbr mapping from assets for stable comparison
        try:
            assets = _load_team_assets()
            abbr_map = {}
            if isinstance(assets, dict):
                for k, v in assets.items():
                    try:
                        abbr_map[str(k).strip().upper()] = str(v.get("abbr") or k).strip().upper()
                    except Exception:
                        continue
            def to_abbr(t):
                s = str(t or "").strip()
                return abbr_map.get(s.upper(), s.upper()) if s else ""
        except Exception:
            def to_abbr(t):
                return str(t or "").strip().upper()

        def collect_games(df_src, seen):
            if df_src is None or df_src.empty:
                return []
            rows = []
            cols = [c for c in ["home_team","away_team"] if c in df_src.columns]
            if len(cols) < 2:
                return []
            for _, r in df_src[cols].dropna(how="any").drop_duplicates().iterrows():
                ht_raw = r.get("home_team"); at_raw = r.get("away_team")
                ht = _norm_team(ht_raw); at = _norm_team(at_raw)
                key = (to_abbr(ht), to_abbr(at))
                if not key[0] or not key[1] or key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "event": f"{at} @ {ht}",
                    "home_team": ht,
                    "away_team": at,
                })
            return rows

        seen = set()
        # Priority 1: weekly schedule for this season/week (ensures started/completed games appear)
        try:
            if games_df is not None and not games_df.empty:
                gdf = games_df.copy()
                # Coerce season/week to numeric for filtering
                if "season" in gdf.columns:
                    gdf["season"] = pd.to_numeric(gdf["season"], errors="coerce")
                if "week" in gdf.columns:
                    gdf["week"] = pd.to_numeric(gdf["week"], errors="coerce")
                if ("season" in gdf.columns and "week" in gdf.columns) and (season_i is not None and week_i is not None):
                    gdf = gdf[(gdf["season"] == season_i) & (gdf["week"] == week_i)]
                games.extend(collect_games(gdf, seen))
        except Exception:
            pass
        # Priority 2: edges_df (has props context)
        games.extend(collect_games(edges_df, seen))
        # Priority 3: bovada raw
        games.extend(collect_games(bov_df, seen))
    except Exception:
        games = []

    # Optional game filter (robust: match by team abbreviations parsed from event/home/away)
    ev_param = request.args.get("event")
    home_param = request.args.get("home_team")
    away_param = request.args.get("away_team")
    all_param = request.args.get("all")
    try:
        show_all = str(all_param).strip().lower() in {"1","true","yes","y"}
    except Exception:
        show_all = False

    # Build abbreviation helpers similar to game-props to make filtering resilient
    def _build_abbr_helpers():
        try:
            assets = _load_team_assets()
            abbr_map = {}
            nick_to_abbr = {}
            if isinstance(assets, dict):
                for full, meta in assets.items():
                    try:
                        ab = str((meta.get('abbr') if isinstance(meta, dict) else None) or full).strip().upper()
                        full_up = str(full).strip().upper()
                        abbr_map[full_up] = ab
                        abbr_map[ab] = ab
                        parts = [p for p in str(full).strip().split() if p]
                        if parts:
                            nick = parts[-1].upper()
                            if nick not in nick_to_abbr:
                                nick_to_abbr[nick] = ab
                    except Exception:
                        continue
            def to_abbr_any(x: object) -> str:
                s = str(x or '').strip()
                if not s:
                    return ''
                u = s.upper()
                if u in abbr_map:
                    return abbr_map[u]
                parts = [p for p in u.split() if p]
                if parts:
                    nick = parts[-1]
                    if nick in nick_to_abbr:
                        return nick_to_abbr[nick]
                return u
            return to_abbr_any
        except Exception:
            def to_abbr_any(x: object) -> str:
                return str(x or '').strip().upper()
            return to_abbr_any

    to_abbr_any = _build_abbr_helpers()

    # Ensure abbreviations columns exist for filtering; derive from home/away when available, else parse event per-row
    try:
        if {"home_team","away_team"}.issubset(edges_df.columns):
            edges_df["__home_abbr"] = edges_df["home_team"].map(to_abbr_any)
            edges_df["__away_abbr"] = edges_df["away_team"].map(to_abbr_any)
        elif "event" in edges_df.columns:
            def _row_home_abbr(ev: object) -> str:
                s = str(ev or '')
                if "@" in s:
                    try:
                        parts = [p.strip() for p in s.split("@", 1)]
                        if len(parts) == 2:
                            return to_abbr_any(parts[1])
                    except Exception:
                        return ''
                return ''
            def _row_away_abbr(ev: object) -> str:
                s = str(ev or '')
                if "@" in s:
                    try:
                        parts = [p.strip() for p in s.split("@", 1)]
                        if len(parts) == 2:
                            return to_abbr_any(parts[0])
                    except Exception:
                        return ''
                return ''
            edges_df["__home_abbr"] = edges_df["event"].map(_row_home_abbr)
            edges_df["__away_abbr"] = edges_df["event"].map(_row_away_abbr)
    except Exception:
        pass

    # Apply filter using abbreviations when a specific game is requested
    try:
        if ev_param and "@" in str(ev_param) and {"__home_abbr","__away_abbr"}.issubset(edges_df.columns):
            parts = [p.strip() for p in str(ev_param).split("@", 1)]
            if len(parts) == 2:
                tgt_away_ab = to_abbr_any(parts[0])
                tgt_home_ab = to_abbr_any(parts[1])
                edges_df = edges_df[(edges_df["__home_abbr"].astype(str) == tgt_home_ab) & (edges_df["__away_abbr"].astype(str) == tgt_away_ab)]
        elif home_param and away_param and {"__home_abbr","__away_abbr"}.issubset(edges_df.columns):
            edges_df = edges_df[(edges_df["__home_abbr"].astype(str) == to_abbr_any(home_param)) & (edges_df["__away_abbr"].astype(str) == to_abbr_any(away_param))]
    except Exception:
        pass

    # If a specific game is selected but edges yielded no rows, fallback to Bovada rows for that game
    bovada_fallback_used = False
    bovada_fallback_source = None  # 'current' | 'archive'
    try:
        want_specific = False
        tgt_home_ab = tgt_away_ab = None
        if ev_param and "@" in str(ev_param):
            parts = [p.strip() for p in str(ev_param).split("@", 1)]
            if len(parts) == 2:
                tgt_away_ab, tgt_home_ab = to_abbr_any(parts[0]), to_abbr_any(parts[1])
                want_specific = True
        elif home_param and away_param:
            tgt_home_ab, tgt_away_ab = to_abbr_any(home_param), to_abbr_any(away_param)
            want_specific = True
        if want_specific and (edges_df is None or edges_df.empty):
            # First try current Bovada file
            bsel = pd.DataFrame()
            if bov_df is not None and not bov_df.empty:
                b = bov_df.copy()
                # Derive abbreviations for filtering
                try:
                    if {"home_team","away_team"}.issubset(b.columns):
                        b["__home_abbr"] = b["home_team"].map(to_abbr_any)
                        b["__away_abbr"] = b["away_team"].map(to_abbr_any)
                    elif "event" in b.columns:
                        def _home_ab(ev):
                            s = str(ev or '')
                            if "@" in s:
                                parts = [p.strip() for p in s.split("@", 1)]
                                if len(parts) == 2:
                                    return to_abbr_any(parts[1])
                            return ''
                        def _away_ab(ev):
                            s = str(ev or '')
                            if "@" in s:
                                parts = [p.strip() for p in s.split("@", 1)]
                                if len(parts) == 2:
                                    return to_abbr_any(parts[0])
                            return ''
                        b["__home_abbr"], b["__away_abbr"] = b["event"].map(_home_ab), b["event"].map(_away_ab)
                except Exception:
                    pass
                # Filter to selected game
                try:
                    bsel = b[(b["__home_abbr"].astype(str) == tgt_home_ab) & (b["__away_abbr"].astype(str) == tgt_away_ab)]
                except Exception:
                    bsel = pd.DataFrame()
                if bsel is not None and not bsel.empty:
                    bovada_fallback_used = True
                    bovada_fallback_source = 'current'
            # If still empty, try archived Bovada snapshot (if present)
            if bsel is None or bsel.empty:
                try:
                    arch_fp = DATA_DIR / f"bovada_player_props_{season_i}_wk{week_i}.archive.csv"
                    if arch_fp.exists():
                        ba = _read_csv_flex(arch_fp)
                    else:
                        ba = pd.DataFrame()
                except Exception:
                    ba = pd.DataFrame()
                if ba is not None and not ba.empty:
                    try:
                        if {"home_team","away_team"}.issubset(ba.columns):
                            ba["__home_abbr"] = ba["home_team"].map(to_abbr_any)
                            ba["__away_abbr"] = ba["away_team"].map(to_abbr_any)
                        elif "event" in ba.columns:
                            def _home_ab(ev):
                                s = str(ev or '')
                                if "@" in s:
                                    parts = [p.strip() for p in s.split("@", 1)]
                                    if len(parts) == 2:
                                        return to_abbr_any(parts[1])
                                return ''
                            def _away_ab(ev):
                                s = str(ev or '')
                                if "@" in s:
                                    parts = [p.strip() for p in s.split("@", 1)]
                                    if len(parts) == 2:
                                        return to_abbr_any(parts[0])
                                return ''
                            ba["__home_abbr"], ba["__away_abbr"] = ba["event"].map(_home_ab), ba["event"].map(_away_ab)
                    except Exception:
                        pass
                    try:
                        bsel = ba[(ba["__home_abbr"].astype(str) == tgt_home_ab) & (ba["__away_abbr"].astype(str) == tgt_away_ab)]
                    except Exception:
                        bsel = pd.DataFrame()
                    if bsel is not None and not bsel.empty:
                        bovada_fallback_used = True
                        bovada_fallback_source = 'archive'
            if bsel is not None and not bsel.empty:
                # Map market -> market_key minimally
                if "market_key" not in bsel.columns and "market" in bsel.columns:
                    mk_map = {
                        "receiving yards": "rec_yards",
                        "receptions": "receptions",
                        "rushing yards": "rush_yards",
                        "passing yards": "pass_yards",
                        "passing tds": "pass_tds",
                        "pass tds": "pass_tds",
                        "pass touchdowns": "pass_tds",
                        "passing attempts": "pass_attempts",
                        "pass attempts": "pass_attempts",
                        "rushing attempts": "rush_attempts",
                        "rush attempts": "rush_attempts",
                        "interceptions": "interceptions",
                        "interceptions thrown": "interceptions",
                        "rush+rec yards": "rush_rec_yards",
                        "rushing + receiving yards": "rush_rec_yards",
                        "rush + rec yards": "rush_rec_yards",
                        "pass+rush yards": "pass_rush_yards",
                        "pass + rush yards": "pass_rush_yards",
                        "passing + rushing yards": "pass_rush_yards",
                        "targets": "targets",
                        "2+ touchdowns": "multi_tds",
                        "anytime td": "any_td",
                        "any time td": "any_td",
                    }
                    try:
                        bsel["market_key"] = bsel["market"].astype(str).str.strip().str.lower().map(mk_map).fillna(bsel["market"].astype(str).str.strip().str.lower())
                    except Exception:
                        pass
                # Ensure price columns present
                for c in ["over_price","under_price"]:
                    if c not in bsel.columns:
                        bsel[c] = pd.NA
                # Keep only relevant columns to mimic edges_df shape
                keep_cols = [c for c in [
                    "player","team","event","home_team","away_team","game_time",
                    "market","market_key","line","over_price","under_price"
                ] if c in bsel.columns]
                bsel = bsel[keep_cols].copy()
                # Use selection as edges_df
                edges_df = bsel
    except Exception:
        pass

    # Enforce team-to-game alignment: keep rows where player's team matches either game team
    try:
        if {"home_team","away_team","team"}.issubset(edges_df.columns) and len(edges_df) > 0:
            try:
                assets = _load_team_assets()
                abbr_map = {}
                nick_to_abbr = {}
                if isinstance(assets, dict):
                    for full, meta in assets.items():
                        try:
                            ab = str((meta.get('abbr') if isinstance(meta, dict) else None) or full).strip().upper()
                            full_up = str(full).strip().upper()
                            abbr_map[full_up] = ab
                            abbr_map[ab] = ab
                            parts = [p for p in str(full).strip().split() if p]
                            if parts:
                                nick = parts[-1].upper()
                                if nick not in nick_to_abbr:
                                    nick_to_abbr[nick] = ab
                        except Exception:
                            continue
                def to_abbr(t):
                    s = str(t or "").strip()
                    if not s:
                        return ""
                    u = s.upper()
                    # Direct abbr or full name
                    if u in abbr_map:
                        return abbr_map[u]
                    # Nickname mapping (e.g., BILLS -> BUF)
                    parts = [p for p in u.split() if p]
                    if parts:
                        nick = parts[-1]
                        if nick in nick_to_abbr:
                            return nick_to_abbr[nick]
                    return u
            except Exception:
                def to_abbr(t):
                    return str(t or "").strip().upper()
            # Compute comparison keys
            ht_abbr = edges_df["home_team"].map(to_abbr)
            at_abbr = edges_df["away_team"].map(to_abbr)
            tm_abbr = edges_df["team"].map(to_abbr)
            mask_keep = (
                tm_abbr.isna() | ht_abbr.isna() | at_abbr.isna() |
                (tm_abbr.eq(ht_abbr)) | (tm_abbr.eq(at_abbr))
            )
            if mask_keep.notna().any():
                edges_df = edges_df[mask_keep.fillna(True)]
    except Exception:
        pass

    # Default now: do not narrow implicitly; return all games unless caller filters.

    # Join minimal player context from predictions (position/team/opponent + projections)
    player_key = None
    try:
        cand_name_cols = [c for c in ["display_name","player","name","player_name"] if c in preds_df.columns]
        if cand_name_cols:
            name_col = cand_name_cols[0]
            preds_df = preds_df.copy()
            # Build normalized keys using name_normalizer (fallback to simple rules if import fails)
            try:
                from nfl_compare.src.name_normalizer import normalize_name_loose as _nm_loose, normalize_alias_init_last as _nm_alias
            except Exception:
                def _nm_loose(s):
                    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())
                def _nm_alias(s):
                    s = str(s or "").strip().lower(); parts = [p for p in s.replace("-"," ").replace("."," ").split() if p];
                    return f"{parts[0][:1]}{''.join(ch for ch in (parts[-1] if parts else '') if ch.isalnum())}" if parts else ""
            preds_df["__key_player"] = preds_df[name_col].astype(str).str.strip().str.lower()
            preds_df["__key_player_loose"] = preds_df[name_col].map(_nm_loose)
            preds_df["__key_player_alias"] = preds_df[name_col].map(_nm_alias)
            pcols_core = ["__key_player","__key_player_loose","__key_player_alias"]
            proj_cols = [c for c in [
                "position","team","opponent",
                # QB
                "pass_attempts","pass_yards","pass_tds","interceptions",
                # RB/WR/TE
                "rush_attempts","rush_yards","rush_tds",
                "targets","receptions","rec_yards","rec_tds",
                # Combos / probabilities (best-effort)
                "rush_rec_yards","pass_rush_yards","any_td_prob",
            ] if c in preds_df.columns]
            # Normalize team to shared abbreviation to improve merge specificity
            try:
                assets = _load_team_assets()
                # Map full name -> abbr (upper); and abbr -> abbr
                full_to_abbr = {str(k): str(v.get("abbr") or k) for k, v in assets.items()} if isinstance(assets, dict) else {}
                abbr_set = {str(v.get("abbr")).upper() for v in assets.values() if isinstance(v, dict) and v.get("abbr")}
                def to_abbr(x: Optional[str]) -> Optional[str]:
                    if x is None or (isinstance(x, float) and pd.isna(x)):
                        return None
                    s = str(x).strip()
                    if not s:
                        return None
                    # If already abbr-like and known
                    if s.upper() in abbr_set:
                        return s.upper()
                    # Try full name mapping
                    ab = full_to_abbr.get(s)
                    return str(ab).upper() if ab else None
                if "team" in preds_df.columns:
                    preds_df["__team_abbr"] = preds_df["team"].map(to_abbr)
                else:
                    preds_df["__team_abbr"] = None
            except Exception:
                preds_df["__team_abbr"] = None
            # Build keys on edges side
            pcol = "player" if "player" in edges_df.columns else edges_df.columns[0]
            # Sanitize player names: strip trailing team tags like (MIA) and any trailing "Total ..." labels
            try:
                def _clean_player_name(s: str) -> str:
                    t = str(s or "")
                    # Remove trailing (TEAM)
                    t = pd.Series([t]).str.replace(r"\s*\([A-Za-z]{2,4}\)\s*$", "", regex=True).iloc[0]
                    # Remove trailing "Total ..." descriptors often present in some book exports
                    t = pd.Series([t]).str.replace(r"\s+total\b.*$", "", regex=True, case=False).iloc[0]
                    return str(t).strip()
                edges_df[pcol] = edges_df[pcol].astype(str).map(_clean_player_name)
            except Exception:
                # Fallback: basic strip
                edges_df[pcol] = edges_df[pcol].astype(str).str.replace(r"\s*\([A-Za-z]{2,4}\)\s*$", "", regex=True).str.strip()
            edges_df["__key_player"] = edges_df[pcol].astype(str).str.strip().str.lower()
            edges_df["__key_player_loose"] = edges_df[pcol].map(_nm_loose)
            edges_df["__key_player_alias"] = edges_df[pcol].map(_nm_alias)
            try:
                if "team" in edges_df.columns:
                    edges_df["__team_abbr"] = edges_df["team"].map(lambda x: str(x).strip().upper() if pd.notna(x) and str(x).strip() else None)
                else:
                    edges_df["__team_abbr"] = None
            except Exception:
                edges_df["__team_abbr"] = None
            # Filter out non-player rows (e.g., quarter/team markets like "1Q BUF/MIA ...")
            try:
                def _looks_like_player_name(s: str) -> bool:
                    t = str(s or "").strip()
                    if not t:
                        return False
                    # Exclude obvious non-player patterns
                    if any(ch.isdigit() for ch in t):
                        return False
                    if ("/" in t) or ("@" in t):
                        return False
                    lt = t.lower()
                    if any(x in lt for x in ["1q","2q","3q","4q","quarter","first quarter","second quarter","first half","second half","1h","2h"]):
                        return False
                    parts = [p for p in t.replace(".", "").replace("'", "").replace("-", " ").split() if p]
                    if len(parts) < 2:
                        return False
                    if parts[0].lower() in {"over","under"}:
                        return False
                    return True
                pcol = "player" if "player" in edges_df.columns else edges_df.columns[0]
                if pcol in edges_df.columns and len(edges_df) > 0:
                    _mask_players = edges_df[pcol].map(_looks_like_player_name)
                    if getattr(_mask_players, 'any', lambda: False)() and (~_mask_players).sum() > 0:
                        edges_df = edges_df[_mask_players].copy()
            except Exception:
                pass
            player_key = "__key_player"
            # Ensure destination columns exist for context fields before fills
            for c in [
                "position","team","opponent",
                "pass_attempts","pass_yards","pass_tds","interceptions",
                "rush_attempts","rush_yards","rush_tds",
                "targets","receptions","rec_yards","rec_tds",
                "rush_rec_yards","pass_rush_yards","any_td_prob",
            ]:
                if c not in edges_df.columns:
                    edges_df[c] = pd.NA
            # Strict join
            try:
                # Prefer joining with team key when available to avoid name collisions
                if "__team_abbr" in edges_df.columns and "__team_abbr" in preds_df.columns and edges_df["__team_abbr"].notna().any():
                    edges_df = edges_df.merge(
                        preds_df[["__key_player", "__team_abbr", *proj_cols]],
                        on=["__key_player", "__team_abbr"],
                        how="left",
                        suffixes=("", "_p"),
                    )
                else:
                    edges_df = edges_df.merge(
                        preds_df[["__key_player", *proj_cols]],
                        on="__key_player",
                        how="left",
                        suffixes=("", "_p"),
                    )
                # Coalesce merged projection/context columns into base cols and drop suffixed ones
                for c in proj_cols:
                    try:
                        c_p = f"{c}_p"
                        if c_p in edges_df.columns:
                            # For context columns, prefer predictions value when available to avoid wrong-team artifacts
                            if c in {"position","team","opponent"}:
                                if c in edges_df.columns:
                                    edges_df[c] = edges_df[c_p].where(edges_df[c_p].notna(), edges_df[c])
                                else:
                                    edges_df[c] = edges_df[c_p]
                                edges_df.drop(columns=[c_p], inplace=True)
                            else:
                                # For numeric projections, keep existing if present; fill from predictions when missing
                                if c in edges_df.columns:
                                    edges_df[c] = edges_df[c].where(edges_df[c].notna(), edges_df[c_p])
                                    edges_df.drop(columns=[c_p], inplace=True)
                                else:
                                    edges_df.rename(columns={c_p: c}, inplace=True)
                    except Exception:
                        pass
                # Cleanup if an earlier merge created _x/_y columns (defensive)
                for c in [
                    "position","team","opponent",
                    "pass_attempts","pass_yards","pass_tds","interceptions",
                    "rush_attempts","rush_yards","rush_tds",
                    "targets","receptions","rec_yards","rec_tds",
                    "rush_rec_yards","pass_rush_yards","any_td_prob",
                ]:
                    cx, cy = f"{c}_x", f"{c}_y"
                    if cx in edges_df.columns and cy in edges_df.columns:
                        try:
                            edges_df[c] = edges_df[cx].where(edges_df[cx].notna(), edges_df[cy])
                            edges_df.drop(columns=[cx, cy], inplace=True)
                        except Exception:
                            # Best-effort: if base missing, rename one of them
                            if c not in edges_df.columns:
                                try:
                                    edges_df.rename(columns={cx: c}, inplace=True)
                                    edges_df.drop(columns=[cy], inplace=True)
                                except Exception:
                                    pass
                # Recompute team abbreviation after potential team override
                try:
                    if "team" in edges_df.columns:
                        edges_df["__team_abbr"] = edges_df["team"].map(lambda x: str(x).strip().upper() if pd.notna(x) and str(x).strip() else None)
                except Exception:
                    pass
            except Exception:
                pass
            # Loose join fill
            try:
                # Trigger fill when numeric projection fields are missing after strict join
                numeric_targets = [c for c in ["pass_yards","rush_yards","rec_yards","receptions","any_td_prob"] if c in preds_df.columns]
                present_proj_cols = [c for c in numeric_targets if c in edges_df.columns]
                if present_proj_cols:
                    missing_mask = ~edges_df[present_proj_cols].notna().any(axis=1)
                else:
                    # No projection cols present yet -> attempt fill for all rows
                    missing_mask = pd.Series([True] * len(edges_df), index=edges_df.index)
                if missing_mask.any():
                    if "__team_abbr" in edges_df.columns and "__team_abbr" in preds_df.columns and edges_df["__team_abbr"].notna().any():
                        fill = preds_df[["__key_player_loose", "__team_abbr", *proj_cols]].rename(columns={"__key_player_loose": "__k2"})
                        edges_df = edges_df.merge(fill, on=["__team_abbr"], left_on=["__key_player_loose","__team_abbr"], right_on=["__k2","__team_abbr"], how="left", suffixes=("", "_b"))
                    else:
                        fill = preds_df[["__key_player_loose", *proj_cols]].rename(columns={"__key_player_loose": "__k2"})
                        edges_df = edges_df.merge(fill, left_on="__key_player_loose", right_on="__k2", how="left", suffixes=("", "_b"))
                    for c in proj_cols:
                        if c in edges_df.columns and f"{c}_b" in edges_df.columns:
                            edges_df[c] = edges_df[c].where(edges_df[c].notna(), edges_df[f"{c}_b"])
                    edges_df = edges_df.drop(columns=[c for c in ["__k2", *[f"{c}_b" for c in proj_cols]] if c in edges_df.columns])
            except Exception:
                pass
            # Alias join fill
            try:
                if present_proj_cols:
                    missing_mask = ~edges_df[present_proj_cols].notna().any(axis=1)
                else:
                    missing_mask = pd.Series([True] * len(edges_df), index=edges_df.index)
                if missing_mask.any():
                    if "__team_abbr" in edges_df.columns and "__team_abbr" in preds_df.columns and edges_df["__team_abbr"].notna().any():
                        fill = preds_df[["__key_player_alias", "__team_abbr", *proj_cols]].rename(columns={"__key_player_alias": "__k3"})
                        edges_df = edges_df.merge(fill, on=["__team_abbr"], left_on=["__key_player_alias","__team_abbr"], right_on=["__k3","__team_abbr"], how="left", suffixes=("", "_c"))
                    else:
                        fill = preds_df[["__key_player_alias", *proj_cols]].rename(columns={"__key_player_alias": "__k3"})
                        edges_df = edges_df.merge(fill, left_on="__key_player_alias", right_on="__k3", how="left", suffixes=("", "_c"))
                    for c in proj_cols:
                        if c in edges_df.columns and f"{c}_c" in edges_df.columns:
                            edges_df[c] = edges_df[c].where(edges_df[c].notna(), edges_df[f"{c}_c"])
                    edges_df = edges_df.drop(columns=[c for c in ["__k3", *[f"{c}_c" for c in proj_cols]] if c in edges_df.columns])
            except Exception:
                pass
            # Ensure core context fields are filled for display grouping and projection selection
            try:
                for tgt_col, key_name in [("position","position"),("team","team"),("opponent","opponent")]:
                    if tgt_col in edges_df.columns and edges_df[tgt_col].isna().any() and key_name in preds_df.columns:
                        miss = edges_df[tgt_col].isna()
                        # Strict fill
                        src1 = preds_df[["__key_player", key_name]].drop_duplicates().rename(columns={"__key_player":"__k1", key_name:"__v1"})
                        edges_df = edges_df.merge(src1, left_on="__key_player", right_on="__k1", how="left")
                        edges_df.loc[miss, tgt_col] = edges_df.loc[miss, tgt_col].where(edges_df.loc[miss, tgt_col].notna(), edges_df.loc[miss, "__v1"])
                        edges_df = edges_df.drop(columns=[c for c in ["__k1","__v1"] if c in edges_df.columns])
                        # Loose fill
                        miss = edges_df[tgt_col].isna()
                        if miss.any():
                            src2 = preds_df[["__key_player_loose", key_name]].drop_duplicates().rename(columns={"__key_player_loose":"__k2", key_name:"__v2"})
                            edges_df = edges_df.merge(src2, left_on="__key_player_loose", right_on="__k2", how="left")
                            edges_df.loc[miss, tgt_col] = edges_df.loc[miss, tgt_col].where(edges_df.loc[miss, tgt_col].notna(), edges_df.loc[miss, "__v2"])
                            edges_df = edges_df.drop(columns=[c for c in ["__k2","__v2"] if c in edges_df.columns])
                            miss = edges_df[tgt_col].isna()
                        # Alias fill
                        if miss.any():
                            src3 = preds_df[["__key_player_alias", key_name]].drop_duplicates().rename(columns={"__key_player_alias":"__k3", key_name:"__v3"})
                            edges_df = edges_df.merge(src3, left_on="__key_player_alias", right_on="__k3", how="left")
                            edges_df.loc[miss, tgt_col] = edges_df.loc[miss, tgt_col].where(edges_df.loc[miss, tgt_col].notna(), edges_df.loc[miss, "__v3"])
                            edges_df = edges_df.drop(columns=[c for c in ["__k3","__v3"] if c in edges_df.columns])
            except Exception:
                pass
            # Apply canonical team/position from predictions by player key to correct wrong-team artifacts
            try:
                canon = preds_df[["__key_player","team","position"]].dropna(subset=["__key_player"]).drop_duplicates()
                edges_df = edges_df.merge(canon.rename(columns={"team":"__canon_team","position":"__canon_pos"}), on="__key_player", how="left")
                if "__canon_team" in edges_df.columns:
                    edges_df["team"] = edges_df["__canon_team"].where(edges_df["__canon_team"].notna(), edges_df["team"])
                if "__canon_pos" in edges_df.columns:
                    edges_df["position"] = edges_df["__canon_pos"].where(edges_df["__canon_pos"].notna(), edges_df["position"])
                edges_df = edges_df.drop(columns=[c for c in ["__canon_team","__canon_pos"] if c in edges_df.columns])
                # Recompute team abbr again after canonical override
                try:
                    if "team" in edges_df.columns:
                        edges_df["__team_abbr"] = edges_df["team"].map(lambda x: str(x).strip().upper() if pd.notna(x) and str(x).strip() else None)
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        player_key = None

    # Group by player for card output
    cards = []
    try:
        # Local light normalizer (letters+digits, lowercase)
        def _nm_loose_local(s):
            try:
                return "".join(ch for ch in str(s or "").lower() if ch.isalnum())
            except Exception:
                return str(s or "").strip().lower()

        def _row_to_play(r: pd.Series) -> dict:
            ev_pct = None
            # Compute EV% generically when available, keyed off selected side
            try:
                side = r.get("rec_side")
                ov = r.get("over_ev"); un = r.get("under_ev")
                ov_ok = (ov is not None and not pd.isna(ov))
                un_ok = (un is not None and not pd.isna(un))
                # Special case: ATD with 0% projection -> skip entirely
                if (r.get("market_key") == "any_td"):
                    try:
                        proj_val = pd.to_numeric(r.get("proj"), errors="coerce")
                        if pd.isna(proj_val) or float(proj_val) == 0.0:
                            return {}
                    except Exception:
                        pass
                if side == "Over" and ov_ok:
                    ev_pct = float(ov) * 100.0
                elif side == "Under" and un_ok:
                    ev_pct = float(un) * 100.0
                elif side is None:
                    # If side not chosen, infer from EVs when possible to drive UI coloring and EV chip
                    if ov_ok and un_ok:
                        if float(ov) >= float(un):
                            side = "Over"; ev_pct = float(ov) * 100.0
                        else:
                            side = "Under"; ev_pct = float(un) * 100.0
                    elif ov_ok and not un_ok:
                        side = "Over"; ev_pct = float(ov) * 100.0
                    elif un_ok and not ov_ok:
                        side = "Under"; ev_pct = float(un) * 100.0
                # If no EV data at all for this row and it's ATD, skip (non-actionable)
                if r.get("market_key") == "any_td" and (not ov_ok and not un_ok):
                    return {}
            except Exception:
                ev_pct = None
            # Projection-vs-line fallback to choose side (for line-based markets) if still None
            try:
                if side is None:
                    mk = r.get("market_key") or r.get("market")
                    mk = (str(mk).strip().lower() if mk is not None else "")
                    line_required = {
                        "rec_yards","rush_yards","pass_yards","receptions",
                        "pass_attempts","rush_attempts","rush_rec_yards","pass_rush_yards","targets"
                    }
                    if mk in line_required:
                        pr = pd.to_numeric(r.get("proj"), errors="coerce")
                        ln = pd.to_numeric(r.get("line"), errors="coerce")
                        if pd.notna(pr) and pd.notna(ln):
                            if float(pr) > float(ln):
                                side = "Over"
                            elif float(pr) < float(ln):
                                side = "Under"
            except Exception:
                pass
            # Derive ladder flag if present in edges CSV
            is_ladder = None
            try:
                v = r.get("is_ladder")
                # Respect pandas NA
                if v is not None and not pd.isna(v):
                    s = str(v).strip().lower()
                    if s in {"1","true","yes","y"}:
                        is_ladder = True
                    elif s in {"0","false","no","n"}:
                        is_ladder = False
                    else:
                        # Fallback: truthiness
                        try:
                            is_ladder = bool(int(s))
                        except Exception:
                            is_ladder = bool(v)
            except Exception:
                is_ladder = None
            play = {
                "market": r.get("market") or r.get("market_key"),
                "line": r.get("line"),
                "proj": r.get("proj"),
                "edge": r.get("edge"),
                "over_price": r.get("over_price"),
                "under_price": r.get("under_price"),
                # Use inferred side (from rec_side or EV fallback) so UI can color prices
                "side": side,
                "ev_pct": ev_pct,
                "is_ladder": is_ladder,
            }
            # Skip non-actionable/invalid plays
            try:
                # Normalize to internal key for validation checks
                raw_mk = r.get("market_key")
                mk = None
                if raw_mk is not None and not (isinstance(raw_mk, float) and pd.isna(raw_mk)):
                    mk = str(raw_mk).strip().lower()
                if not mk:
                    # Map display market -> internal key
                    mk_map_local = {
                        "receiving yards": "rec_yards",
                        "receptions": "receptions",
                        "rushing yards": "rush_yards",
                        "passing yards": "pass_yards",
                        "passing tds": "pass_tds",
                        "pass tds": "pass_tds",
                        "pass touchdowns": "pass_tds",
                        "passing attempts": "pass_attempts",
                        "pass attempts": "pass_attempts",
                        "rushing attempts": "rush_attempts",
                        "rush attempts": "rush_attempts",
                        "interceptions": "interceptions",
                        "interceptions thrown": "interceptions",
                        "rush+rec yards": "rush_rec_yards",
                        "rushing + receiving yards": "rush_rec_yards",
                        "rush + rec yards": "rush_rec_yards",
                        "pass+rush yards": "pass_rush_yards",
                        "pass + rush yards": "pass_rush_yards",
                        "passing + rushing yards": "pass_rush_yards",
                        "targets": "targets",
                        "2+ touchdowns": "multi_tds",
                        "anytime td": "any_td",
                        "any time td": "any_td",
                    }
                    disp = str(r.get("market") or "").strip().lower()
                    mk = mk_map_local.get(disp, disp)
                has_side = bool(play.get("side"))
                has_line = pd.notna(play.get("line")) if play.get("line") is not None else False
                has_price = (pd.notna(play.get("over_price")) if play.get("over_price") is not None else False) or (pd.notna(play.get("under_price")) if play.get("under_price") is not None else False)
                # Markets that require a numeric line to be meaningful (yardage, receptions, attempts, combos, targets)
                line_required = {
                    "rec_yards","rush_yards","pass_yards","receptions",
                    "pass_attempts","rush_attempts","rush_rec_yards","pass_rush_yards","targets"
                }
                # EV-driven markets can omit a line
                ev_allowed = {"any_td","interceptions","pass_tds","multi_tds"}
                if mk in line_required and not has_line:
                    return {}
                # If no side, no line, and only prices exist, skip (price-only without target)
                if not has_side and not has_line and not has_price:
                    return {}
            except Exception:
                pass
            return play

        # Determine grouping key: prefer normalized player key and team abbr; avoid including position to prevent split cards
        gcols = []
        if "__key_player_loose" in edges_df.columns:
            gcols.append("__key_player_loose")
        elif "__key_player" in edges_df.columns:
            gcols.append("__key_player")
        elif "player" in edges_df.columns:
            gcols.append("player")
        if "__team_abbr" in edges_df.columns:
            gcols.append("__team_abbr")
        elif "team" in edges_df.columns:
            gcols.append("team")
        if not gcols:
            gcols = [c for c in ["player"] if c in edges_df.columns]
        for _, g in edges_df.groupby(gcols, dropna=False):
            head = g.iloc[0]
            # Choose the most frequent display name within the group as the card title
            try:
                if "player" in g.columns and g["player"].notna().any():
                    name_counts = g["player"].dropna().astype(str).value_counts()
                    disp_name = name_counts.index[0] if not name_counts.empty else head.get("player")
                else:
                    disp_name = head.get("player")
            except Exception:
                disp_name = head.get("player")
            plays = []
            for _, r in g.iterrows():
                try:
                    p = _row_to_play(r)
                    # _row_to_play may return an empty dict to signal skip
                    if p:
                        plays.append(p)
                except Exception:
                    continue
            # Deduplicate identical plays to avoid repeated lines (e.g., duplicated ATD entries)
            try:
                seen = set()
                unique = []
                for p in plays:
                    def norm(v):
                        try:
                            return None if v is None or (isinstance(v, float) and pd.isna(v)) else v
                        except Exception:
                            return v
                    key = (
                        norm(p.get("market")),
                        norm(p.get("line")),
                        norm(p.get("over_price")),
                        norm(p.get("under_price")),
                        norm(p.get("side")),
                        norm(p.get("ev_pct")),
                        norm(p.get("is_ladder")),
                    )
                    if key not in seen:
                        seen.add(key)
                        unique.append(p)
                plays = unique
            except Exception:
                pass
            # Sort plays: EV desc (if available), else abs(edge) desc
            try:
                plays.sort(key=lambda x: (x.get("ev_pct") if x.get("ev_pct") is not None else (abs(x.get("edge")) if x.get("edge") is not None else -9999)), reverse=True)
            except Exception:
                pass
            # Projections bundle (best-effort)
            proj_bundle = {}
            for k in [
                "pass_attempts","pass_yards","pass_tds","interceptions",
                "rush_attempts","rush_yards","rush_tds",
                "targets","receptions","rec_yards","rec_tds",
                # Combos and derived
                "rush_rec_yards","pass_rush_yards",
                # Probabilities last
                "any_td_prob",
            ]:
                if k in edges_df.columns:
                    try:
                        v = head.get(k)
                        proj_bundle[k] = (float(v) if v is not None and not pd.isna(v) else None)
                    except Exception:
                        proj_bundle[k] = None
            # If no actionable plays and no meaningful projections, skip the card
            try:
                has_proj = any(
                    (proj_bundle.get(k) is not None and (proj_bundle.get(k) != 0.0 if k != "any_td_prob" else proj_bundle.get(k) > 0.0))
                    for k in ["pass_yards","rush_yards","rec_yards","receptions","any_td_prob"]
                )
                if not plays and not has_proj:
                    continue
            except Exception:
                pass
            # Canonical event label for consistent UI
            try:
                # Prefer normalized team names when present
                _home = head.get("home_team")
                _away = head.get("away_team")
                try:
                    from nfl_compare.src.team_normalizer import normalize_team_name as _norm_team
                except Exception:
                    def _norm_team(x):
                        return str(x or "").strip()
                ev_lbl = None
                if _home and _away:
                    ev_lbl = f"{_norm_team(_away)} @ {_norm_team(_home)}"
            except Exception:
                ev_lbl = head.get("event")

            cards.append({
                "player": disp_name,
                "position": head.get("position"),
                "team": head.get("team"),
                "opponent": head.get("opponent"),
                "home_team": head.get("home_team"),
                "away_team": head.get("away_team"),
                "event": ev_lbl or head.get("event"),
                "projections": proj_bundle,
                "plays": plays,
            })
    except Exception:
        cards = []

    # Collapse accidental duplicate cards for the same player when team/position are identical
    try:
        if cards:
            # Canonical team/position mapping from predictions at card level as a final normalization
            try:
                # Build map from predictions
                _canon_map_team = {}
                _canon_map_pos = {}
                _canon_map_name = {}
                try:
                    # preds_df may be out of scope if prior block failed; guard defensively
                    _preds = preds_df if 'preds_df' in locals() else pd.DataFrame()
                except Exception:
                    _preds = pd.DataFrame()
                if _preds is not None and not _preds.empty:
                    # Prefer display name column if present for unification
                    name_col = None
                    for nc in ["display_name","player","name","player_name"]:
                        if nc in _preds.columns:
                            name_col = nc; break
                    # Register canonical info under multiple key variants
                    cols = [c for c in ["__key_player","__key_player_loose","__key_player_alias"] if c in _preds.columns]
                    if cols:
                        for _, rr in _preds[[*cols, "team", "position"] + ([name_col] if name_col else [])].dropna(subset=[cols[0]]).drop_duplicates().iterrows():
                            disp = str(rr.get(name_col) if name_col else "").strip() or None
                            for kc in cols:
                                keyv = str(rr.get(kc) or "").strip().lower()
                                if not keyv:
                                    continue
                                if keyv not in _canon_map_team:
                                    _canon_map_team[keyv] = rr.get("team")
                                if keyv not in _canon_map_pos:
                                    _canon_map_pos[keyv] = rr.get("position")
                                if disp and keyv not in _canon_map_name:
                                    _canon_map_name[keyv] = disp
                for c in cards:
                    try:
                        pk_raw = str(c.get("player") or "").strip().lower()
                        pk_loose = _nm_loose_local(pk_raw)
                        # Try exact, then loose
                        ct = _canon_map_team.get(pk_raw)
                        cp = _canon_map_pos.get(pk_raw)
                        dn = _canon_map_name.get(pk_raw)
                        if ct is None and pk_loose:
                            ct = _canon_map_team.get(pk_loose, ct)
                        if cp is None and pk_loose:
                            cp = _canon_map_pos.get(pk_loose, cp)
                        if not dn and pk_loose:
                            dn = _canon_map_name.get(pk_loose)
                        if ct:
                            c["team"] = ct
                        if cp:
                            c["position"] = cp
                        if dn:
                            c["player"] = dn
                    except Exception:
                        continue
            except Exception:
                pass
            # Build team abbr map to create stable merge key
            try:
                assets = _load_team_assets()
                abbr_lookup = {}
                if isinstance(assets, dict):
                    for k, v in assets.items():
                        try:
                            abbr_lookup[str(k).upper()] = str(v.get("abbr") or k).upper()
                        except Exception:
                            continue
                def to_abbr(team: Optional[str]) -> str:
                    if team is None:
                        return ""
                    s = str(team).strip()
                    if not s:
                        return ""
                    up = s.upper()
                    return abbr_lookup.get(up, up)
            except Exception:
                def to_abbr(team: Optional[str]) -> str:
                    return str(team or "").strip().upper()

            merged = {}
            for c in cards:
                pkey = _nm_loose_local(c.get("player"))
                tkey = to_abbr(c.get("team"))
                key = (pkey, tkey)
                if key not in merged:
                    merged[key] = c
                else:
                    # Merge plays, dedup by (market,line,side,price)
                    exist = merged[key].get("plays", [])
                    combo = {(p.get("market"), p.get("line"), p.get("side"), p.get("over_price"), p.get("under_price")) for p in exist}
                    for p in (c.get("plays", []) or []):
                        k = (p.get("market"), p.get("line"), p.get("side"), p.get("over_price"), p.get("under_price"))
                        if k not in combo:
                            exist.append(p); combo.add(k)
                    merged[key]["plays"] = exist
                    # Prefer projections with more non-null fields
                    def proj_score(d):
                        return sum(1 for k in ["pass_yards","rush_yards","rec_yards","receptions","any_td_prob"] if (d or {}).get(k) is not None)
                    if proj_score(c.get("projections")) > proj_score(merged[key].get("projections")):
                        merged[key]["projections"] = c.get("projections")
                    # If event/home/away missing, fill from the other
                    for fld in ["event","home_team","away_team","opponent"]:
                        if not merged[key].get(fld) and c.get(fld):
                            merged[key][fld] = c.get(fld)
            cards = list(merged.values())
    except Exception:
        pass

    # Synthesize projection-only cards for schedule games that lack edges, so all games show player cards
    try:
        # Reuse robust abbreviation helper built earlier
        to_abbr = to_abbr_any
        # Determine if a specific game was requested; if so, restrict synthesis to those two teams
        selected_abbrs = None  # type: Optional[set]
        try:
            if ev_param and "@" in str(ev_param):
                parts = [p.strip() for p in str(ev_param).split("@", 1)]
                if len(parts) == 2:
                    selected_abbrs = {to_abbr(parts[0]), to_abbr(parts[1])}
            elif home_param and away_param:
                selected_abbrs = {to_abbr(home_param), to_abbr(away_param)}
        except Exception:
            selected_abbrs = None
        # Build schedule map: team_abbr -> (home_team, away_team, event)
        sched_map: Dict[str, Tuple[str, str, str]] = {}
        try:
            gdf = games_df.copy() if (games_df is not None and not games_df.empty) else pd.DataFrame()
            if not gdf.empty:
                if "season" in gdf.columns:
                    gdf["season"] = pd.to_numeric(gdf["season"], errors="coerce")
                if "week" in gdf.columns:
                    gdf["week"] = pd.to_numeric(gdf["week"], errors="coerce")
                if ("season" in gdf.columns and "week" in gdf.columns) and (season_i is not None and week_i is not None):
                    gdf = gdf[(gdf["season"] == season_i) & (gdf["week"] == week_i)]
                if not gdf.empty and {"home_team","away_team"}.issubset(gdf.columns):
                    for _, rr in gdf[["home_team","away_team"]].dropna(how='any').drop_duplicates().iterrows():
                        ht = str(rr.get("home_team") or "").strip(); at = str(rr.get("away_team") or "").strip()
                        if not ht or not at:
                            continue
                        ev = f"{at} @ {ht}"
                        sched_map[to_abbr(ht)] = (ht, at, ev)
                        sched_map[to_abbr(at)] = (ht, at, ev)
        except Exception:
            pass
        # Existing keys to avoid duplicates
        exist_keys = set()
        def _nm_loose_local2(s: Optional[str]) -> str:
            return "" if s is None else "".join(ch for ch in str(s).lower() if ch.isalnum())
        for c in cards:
            try:
                pk = _nm_loose_local2(c.get("player")); tb = to_abbr(c.get("team"))
                exist_keys.add((pk, tb))
            except Exception:
                continue
        # Ensure preds_df has name keys; if not, build minimal
        try:
            if "__key_player_loose" not in preds_df.columns:
                name_col = next((c for c in ["display_name","player","name","player_name"] if c in preds_df.columns), None)
                if name_col:
                    preds_df["__key_player_loose"] = preds_df[name_col].astype(str).str.replace("[^a-z0-9]","", regex=True).str.lower()
        except Exception:
            pass
        # Team abbr on preds
        if "__team_abbr" not in preds_df.columns and "team" in preds_df.columns:
            try:
                preds_df["__team_abbr"] = preds_df["team"].map(lambda x: to_abbr(x))
            except Exception:
                preds_df["__team_abbr"] = None
        # Iterate preds and add missing cards for schedule teams
        add_cards = []
        try:
            tcol = "__team_abbr" if "__team_abbr" in preds_df.columns else ("team" if "team" in preds_df.columns else None)
            ncol = "display_name" if "display_name" in preds_df.columns else ("player" if "player" in preds_df.columns else ("name" if "name" in preds_df.columns else None))
            if tcol and ncol and not preds_df.empty:
                for _, rr in preds_df.iterrows():
                    try:
                        tb = to_abbr(rr.get(tcol))
                        if not tb or tb not in sched_map:
                            continue
                        # If a game filter is active, only synthesize for those two teams
                        if selected_abbrs is not None and tb not in selected_abbrs:
                            continue
                        pk = _nm_loose_local2(rr.get(ncol))
                        if (pk, tb) in exist_keys:
                            continue
                        ht, at, ev = sched_map[tb]
                        projections = {}
                        for k in [
                            "pass_attempts","pass_yards","pass_tds","interceptions",
                            "rush_attempts","rush_yards","rush_tds",
                            "targets","receptions","rec_yards","rec_tds",
                            "rush_rec_yards","pass_rush_yards","any_td_prob",
                        ]:
                            if k in preds_df.columns:
                                v = rr.get(k)
                                try:
                                    projections[k] = (float(v) if v is not None and not pd.isna(v) else None)
                                except Exception:
                                    projections[k] = None
                        # Require at least some projection to avoid empty noise
                        has_proj = any((projections.get(k) not in (None, 0.0)) for k in ["pass_yards","rush_yards","rec_yards","receptions"]) or (projections.get("any_td_prob") not in (None, 0.0))
                        if not has_proj:
                            continue
                        add_cards.append({
                            "player": rr.get(ncol),
                            "position": rr.get("position"),
                            "team": rr.get("team"),
                            "opponent": rr.get("opponent"),
                            "home_team": ht,
                            "away_team": at,
                            "event": ev,
                            "projections": projections,
                            "plays": [],
                        })
                        exist_keys.add((pk, tb))
                    except Exception:
                        continue
        except Exception:
            add_cards = []
        if add_cards:
            cards.extend(add_cards)
            # Second-pass merge: synthesized cards may duplicate existing player/team cards.
            try:
                def _nm2(s: object) -> str:
                    try:
                        return "".join(ch for ch in str(s or "").lower() if ch.isalnum())
                    except Exception:
                        return str(s or "").strip().lower()
                def _tabbr(team: object) -> str:
                    try:
                        return to_abbr(team)  # reuse helper from above
                    except Exception:
                        return str(team or "").strip().upper()
                merged2 = {}
                for c in cards:
                    try:
                        key = (_nm2(c.get("player")), _tabbr(c.get("team")))
                        if key not in merged2:
                            merged2[key] = c
                        else:
                            exist = merged2[key]
                            # Merge plays (dedup by market/line/side/prices)
                            try:
                                ep = exist.get("plays") or []
                                np = c.get("plays") or []
                                combo = {(p.get("market"), p.get("line"), p.get("side"), p.get("over_price"), p.get("under_price")) for p in ep}
                                for p in np:
                                    k = (p.get("market"), p.get("line"), p.get("side"), p.get("over_price"), p.get("under_price"))
                                    if k not in combo:
                                        ep.append(p); combo.add(k)
                                exist["plays"] = ep
                            except Exception:
                                pass
                            # Prefer projections with more non-null fields
                            def _proj_score(d):
                                try:
                                    return sum(1 for k in [
                                        "pass_yards","rush_yards","rec_yards","receptions","any_td_prob"
                                    ] if (d or {}).get(k) is not None)
                                except Exception:
                                    return 0
                            try:
                                if _proj_score(c.get("projections")) > _proj_score(exist.get("projections")):
                                    exist["projections"] = c.get("projections")
                            except Exception:
                                pass
                            # Fill missing meta fields
                            for fld in ["event","home_team","away_team","opponent","position"]:
                                if not exist.get(fld) and c.get(fld):
                                    exist[fld] = c.get(fld)
                    except Exception:
                        # If any issue occurs merging one card, skip it and continue
                        continue
                cards = list(merged2.values())
            except Exception:
                pass
    except Exception:
        pass

    # Attach team assets (logo) when available on server side for convenience
    try:
        assets = _load_team_assets()
        abbr_map = {k: (v.get('abbr') or k) for k, v in assets.items()}
        def logo_for(team: Optional[str]) -> Optional[str]:
            if not team:
                return None
            a = assets.get(str(team), {})
            if a.get("logo"):
                return a.get("logo")
            ab = a.get("abbr") or str(team)
            espn_map = {"WAS": "wsh"}
            code = espn_map.get(ab.upper(), ab.lower())
            return f"https://a.espncdn.com/i/teamlogos/nfl/500/{code}.png"
        for c in cards:
            c["team_logo"] = logo_for(c.get("team"))
            # If opponent provided, attach opponent logo for symmetry
            c["opponent_logo"] = logo_for(c.get("opponent"))
    except Exception:
        pass

    # Attach player headshots using nfl_data_py rosters -> ESPN headshot URL (best-effort)
    try:
        # Simple cache keyed by season to avoid repeated loads
        try:
            _HEADSHOT_CACHE  # type: ignore  # noqa: F401
        except Exception:
            _HEADSHOT_CACHE = {}  # type: ignore
        maps = None
        try:
            maps = _HEADSHOT_CACHE.get(season_i)  # type: ignore
        except Exception:
            maps = None
        if maps is None:
            maps = {"by_key": {}, "by_name": {}, "by_alias_key": {}, "by_alias": {}}
            try:
                import nfl_data_py as _nfl  # type: ignore
                ros = _nfl.import_seasonal_rosters([int(season_i)])
            except Exception:
                ros = None
            if ros is not None and not ros.empty:
                # Column picks
                # Prefer 'player_name' for matching (most stable with props names), fallback to other roster labels
                name_col = next((c for c in [
                    "player_name","full_name","display_name","player_display_name","name","gsis_name","football_name"
                ] if c in ros.columns), None)
                team_col = next((c for c in [
                    "team","recent_team","team_abbr","club_code"
                ] if c in ros.columns), None)
                headshot_col = "headshot_url" if "headshot_url" in ros.columns else None
                espn_col = "espn_id" if "espn_id" in ros.columns else ("esb_id" if "esb_id" in ros.columns else None)
                pfr_col = "pfr_id" if "pfr_id" in ros.columns else None
                if name_col:
                    # Team abbr helper from assets
                    try:
                        assets = _load_team_assets()
                        abbr_lookup = {k: (v.get('abbr') or k) for k, v in assets.items()} if isinstance(assets, dict) else {}
                        def to_abbr(team: Optional[str]) -> str:
                            if not team:
                                return ""
                            s = str(team).strip(); up = s.upper()
                            return str(abbr_lookup.get(up, up)).upper()
                    except Exception:
                        def to_abbr(team: Optional[str]) -> str:
                            return str(team or "").strip().upper()
                    # Name normalization
                    try:
                        from nfl_compare.src.name_normalizer import normalize_name_loose as _nm_loose
                    except Exception:
                        def _nm_loose(s):
                            return "" if s is None else "".join(ch for ch in str(s).lower() if ch.isalnum())
                    # Alias normalization (first-initial + last)
                    try:
                        from nfl_compare.src.name_normalizer import normalize_alias_init_last as _nm_alias
                    except Exception:
                        def _nm_alias(s):
                            s = str(s or "").strip().lower()
                            parts = [p for p in s.replace("-"," ").replace("."," ").split() if p]
                            return (parts[0][:1] + ''.join(ch for ch in (parts[-1] if parts else '') if ch.isalnum())) if parts else ""
                    for _, rr in ros.iterrows():
                        try:
                            nm = _nm_loose(rr.get(name_col));
                            if not nm:
                                continue
                            ab = to_abbr(rr.get(team_col)) if team_col else ""
                            al = _nm_alias(rr.get(name_col))
                            url = None
                            if headshot_col:
                                u = rr.get(headshot_col)
                                if isinstance(u, str) and u.strip():
                                    url = u.strip()
                            if url is None and espn_col:
                                eid = rr.get(espn_col)
                                eid_str = str(eid).strip() if eid is not None else ""
                                if eid_str and eid_str.lower() != 'nan':
                                    url = f"https://a.espncdn.com/i/headshots/nfl/players/full/{eid_str}.png"
                            if not url and pfr_col:
                                pid = rr.get(pfr_col)
                                pid_str = str(pid).strip() if pid is not None else ""
                                if pid_str and pid_str.lower() != 'nan':
                                    url = f"https://www.pro-football-reference.com/req/2017/images/headshots/{pid_str}.jpg"
                            if not url:
                                continue
                            key = (nm, str(ab).upper())
                            if key not in maps["by_key"]:
                                maps["by_key"][key] = url
                            if nm not in maps["by_name"]:
                                maps["by_name"][nm] = url
                            if al:
                                ak = (al, str(ab).upper())
                                if ak not in maps["by_alias_key"]:
                                    maps["by_alias_key"][ak] = url
                                if al not in maps["by_alias"]:
                                    maps["by_alias"][al] = url
                        except Exception:
                            continue
            try:
                _HEADSHOT_CACHE[season_i] = maps  # type: ignore
            except Exception:
                pass
        if maps and cards:
            # Normalizer and team abbr helper
            try:
                from nfl_compare.src.name_normalizer import normalize_name_loose as _nm_loose
            except Exception:
                def _nm_loose(s):
                    return "" if s is None else "".join(ch for ch in str(s).lower() if ch.isalnum())
            try:
                from nfl_compare.src.name_normalizer import normalize_alias_init_last as _nm_alias
            except Exception:
                def _nm_alias(s):
                    s = str(s or "").strip().lower()
                    parts = [p for p in s.replace("-"," ").replace("."," ").split() if p]
                    return (parts[0][:1] + ''.join(ch for ch in (parts[-1] if parts else '') if ch.isalnum())) if parts else ""
            try:
                assets = _load_team_assets()
                abbr_lookup = {k: (v.get('abbr') or k) for k, v in assets.items()} if isinstance(assets, dict) else {}
                def to_abbr(team: Optional[str]) -> str:
                    if not team:
                        return ""
                    s = str(team).strip(); up = s.upper()
                    return str(abbr_lookup.get(up, up)).upper()
            except Exception:
                def to_abbr(team: Optional[str]) -> str:
                    return str(team or "").strip().upper()
            for c in cards:
                try:
                    if c.get("player_photo"):
                        continue
                    pk = _nm_loose(c.get("player"))
                    ak = _nm_alias(c.get("player"))
                    tb = to_abbr(c.get("team"))
                    url = (
                        maps.get("by_key", {}).get((pk, tb)) or
                        maps.get("by_name", {}).get(pk) or
                        (maps.get("by_alias_key", {}).get((ak, tb)) if ak else None) or
                        (maps.get("by_alias", {}).get(ak) if ak else None)
                    )
                    # As a final fallback: try a roster last-name search (team-scoped first)
                    if not url:
                        try:
                            import nfl_data_py as _nfl  # type: ignore
                            ros = _nfl.import_seasonal_rosters([int(season_i)])
                            if ros is not None and not ros.empty:
                                name_col = next((cc for cc in [
                                    "player_name","full_name","display_name","player_display_name","name","gsis_name","football_name"
                                ] if cc in ros.columns), None)
                                team_col = next((cc for cc in ["team","recent_team","team_abbr","club_code"] if cc in ros.columns), None)
                                headshot_col = "headshot_url" if "headshot_url" in ros.columns else None
                                espn_col = "espn_id" if "espn_id" in ros.columns else ("esb_id" if "esb_id" in ros.columns else None)
                                pfr_col = "pfr_id" if "pfr_id" in ros.columns else None
                                pl = str(c.get("player") or "").strip()
                                last = pl.split(" ")[-1] if pl else ""
                                cand = ros
                                if team_col and tb:
                                    cand = cand[cand[team_col].astype(str).str.upper() == tb]
                                if name_col and last:
                                    cand = cand[cand[name_col].astype(str).str.contains(last, case=False, na=False)]
                                if cand is not None and not cand.empty:
                                    rr = cand.iloc[0]
                                    u = None
                                    if headshot_col:
                                        v = rr.get(headshot_col)
                                        if isinstance(v, str) and v.strip():
                                            u = v.strip()
                                    if u is None and espn_col:
                                        eid = rr.get(espn_col)
                                        eid_str = str(eid).strip() if eid is not None else ""
                                        if eid_str and eid_str.lower() != 'nan':
                                            u = f"https://a.espncdn.com/i/headshots/nfl/players/full/{eid_str}.png"
                                    if u is None and pfr_col:
                                        pid = rr.get(pfr_col)
                                        pid_str = str(pid).strip() if pid is not None else ""
                                        if pid_str and pid_str.lower() != 'nan':
                                            u = f"https://www.pro-football-reference.com/req/2017/images/headshots/{pid_str}.jpg"
                                    url = u
                        except Exception:
                            pass
                    if url:
                        c["player_photo"] = url
                except Exception:
                    continue
    except Exception:
        pass

    # JSON clean-up: coerce pandas/NumPy NA/NaT and non-finite numbers to None
    def _js(obj):
        try:
            import numpy as _np  # type: ignore
            import pandas as _pd  # type: ignore
        except Exception:
            _np = None
            _pd = None
        try:
            # None and simple short-circuits
            if obj is None:
                return None
            # Pandas NA/NaT
            try:
                if _pd is not None:
                    if obj is getattr(_pd, 'NA', object()) or obj is getattr(_pd, 'NaT', object()):
                        return None
            except Exception:
                pass
            # NumPy scalars and NaN/Inf
            if _np is not None and isinstance(obj, (_np.floating, _np.integer)):
                val = float(obj) if isinstance(obj, _np.floating) else int(obj)
                if isinstance(val, float) and not math.isfinite(val):
                    return None
                return val
            # Native floats
            if isinstance(obj, float):
                return obj if math.isfinite(obj) else None
            # Containers
            if isinstance(obj, dict):
                return {k: _js(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_js(x) for x in obj]
            # Pandas Timestamp -> isoformat
            if _pd is not None and isinstance(obj, getattr(_pd, 'Timestamp', tuple())):
                try:
                    return obj.isoformat()
                except Exception:
                    return str(obj)
            # Other scalars: if pandas NA-like, coerce via pandas.isna
            try:
                if _pd is not None and _pd.isna(obj):
                    return None
            except Exception:
                pass
            return obj
        except Exception:
            return None

    _payload = {
        "season": season_i,
        "week": week_i,
        "games": games,
        "rows": len(cards),
        "data": cards,
        "source": str(edges_fp_used),
        "used_bovada_fallback": bovada_fallback_used,
        "bovada_fallback_source": bovada_fallback_source,
    }
    try:
        _payload = _js(_payload)
    except Exception:
        # Best-effort: if sanitization fails, fall back to minimal safe structure
        _payload = {"season": season_i, "week": week_i, "games": [], "rows": 0, "data": []}
    return jsonify(_payload)


@app.route("/props/recommendations")
def props_recommendations_page():
    """Simple UI for props recommendations with week and game filters."""
    return render_template("props_recommendations.html")


# Backward-compat/typo alias: "/props/recomendations" -> "/props/recommendations"
@app.route("/props/recomendations")
def props_recommendations_typo():
    from flask import redirect
    return redirect("/props/recommendations", code=302)


@app.route("/reconciliation")
def reconciliation_page():
    return render_template("reconciliation.html")

@app.route("/game-props/recommendations")
def game_props_recommendations_page():
    return render_template("game_props_recommendations.html")


@app.route("/api/game-props/recommendations")
def api_game_props_recommendations():
    """Return game-level props recommendations from precomputed edges CSV.

    Query params:
      - season, week (optional; inferred if omitted)
      - home_team, away_team, event (optional filters)
    Response:
      {
        season, week,
        games: [{event, home_team, away_team}],
        rows: N,
        data: [
          { event, home_team, away_team, market_key, team_side, line, side, ev_units, price_home, price_away, over_price, under_price }
        ]
      }
    """
    try:
        season_q = request.args.get("season")
        week_q = request.args.get("week")
        season_i = int(season_q) if season_q else None
        week_i = int(week_q) if week_q else None
    except Exception:
        season_i, week_i = None, None
    pred_df = _load_predictions()
    games_df = _load_games()
    if season_i is None or week_i is None:
        try:
            src = games_df if (games_df is not None and not games_df.empty) else pred_df
            inferred = _infer_current_season_week(src) if (src is not None and not src.empty) else None
            if inferred is not None:
                if season_i is None:
                    season_i = int(inferred[0])
                if week_i is None:
                    week_i = int(inferred[1])
            else:
                if season_i is None and src is not None and not src.empty and 'season' in src.columns and not src['season'].isna().all():
                    season_i = int(src['season'].max())
                if week_i is None:
                    week_i = 1
        except Exception:
            if week_i is None:
                week_i = 1

    # Load edges CSV
    edges_fp = DATA_DIR / f"edges_game_props_{season_i}_wk{week_i}.csv"
    try:
        df = pd.read_csv(edges_fp) if edges_fp.exists() else pd.DataFrame()
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        return jsonify({"season": season_i, "week": week_i, "games": [], "rows": 0, "data": [], "note": f"No edges file: {edges_fp}"})

    # Optional filters
    ev_param = request.args.get("event")
    home_param = request.args.get("home_team")
    away_param = request.args.get("away_team")
    # Hide core markets (ML, spread, total) by default to avoid duplication with main cards
    deriv_flag = str(request.args.get("derivatives_only", "1")).strip().lower() in {"1","true","yes","y"}
    # Keep a copy for fallback when derivatives-only makes a selected game empty
    df_all = df.copy()

    # Normalize team names to abbreviations to make filtering resilient (handles 'Bills' vs 'Buffalo Bills')
    try:
        assets = _load_team_assets()
        abbr_map = {}
        nick_to_abbr = {}
        if isinstance(assets, dict):
            for full, meta in assets.items():
                try:
                    ab = str(meta.get('abbr') or full).strip().upper()
                    full_up = str(full).strip().upper()
                    abbr_map[full_up] = ab
                    abbr_map[ab] = ab
                    # Also map nickname (last token) if unique
                    parts = [p for p in str(full).strip().split() if p]
                    if parts:
                        nick = parts[-1].upper()
                        # If not set or same, map
                        if nick not in nick_to_abbr:
                            nick_to_abbr[nick] = ab
                except Exception:
                    continue
        def to_abbr_any(x: object) -> str:
            s = str(x or '').strip()
            if not s:
                return ''
            u = s.upper()
            # Direct abbr or full name
            if u in abbr_map:
                return abbr_map[u]
            # Try nickname
            parts = [p for p in u.split() if p]
            if parts:
                nick = parts[-1]
                if nick in nick_to_abbr:
                    return nick_to_abbr[nick]
            return u
        if not df.empty and {'home_team','away_team'}.issubset(df.columns):
            df['__home_abbr'] = df['home_team'].map(to_abbr_any)
            df['__away_abbr'] = df['away_team'].map(to_abbr_any)
            df_all['__home_abbr'] = df_all['home_team'].map(to_abbr_any)
            df_all['__away_abbr'] = df_all['away_team'].map(to_abbr_any)
    except Exception:
        pass

    try:
        if deriv_flag and "market_key" in df.columns:
            core = {"moneyline","spread","total"}
            df = df[~df["market_key"].astype(str).str.lower().isin(core)]
        if ev_param and {"__home_abbr","__away_abbr"}.issubset(df.columns):
            # Parse canonical label "Away @ Home" and filter via abbreviations
            if "@" in ev_param:
                parts = [p.strip() for p in ev_param.split("@", 1)]
                if len(parts) == 2:
                    at, ht = parts[0], parts[1]
                    at_ab = to_abbr_any(at)
                    ht_ab = to_abbr_any(ht)
                    df = df[(df["__home_abbr"].astype(str) == ht_ab) & (df["__away_abbr"].astype(str) == at_ab)]
        elif home_param and away_param:
            # Filter via abbreviations when possible
            if {'__home_abbr','__away_abbr'}.issubset(df.columns):
                df = df[(df["__home_abbr"].astype(str) == to_abbr_any(home_param)) & (df["__away_abbr"].astype(str) == to_abbr_any(away_param))]
            else:
                df = df[(df["home_team"].astype(str) == str(home_param)) & (df["away_team"].astype(str) == str(away_param))]
    except Exception:
        pass

    # If derivatives_only removed everything (no filters applied), fallback to include core markets globally
    try:
        if deriv_flag and (df is None or df.empty) and (df_all is not None and not df_all.empty):
            df = df_all.copy()
            fallback_used = True
    except Exception:
        pass

    # If a specific game is selected and derivatives_only yields no rows, fallback to include core markets for that game
    fallback_used = False
    try:
        want_specific = False
        tgt_home = tgt_away = None
        if ev_param and "@" in str(ev_param):
            parts = [p.strip() for p in str(ev_param).split("@", 1)]
            if len(parts) == 2:
                tgt_away, tgt_home = parts[0], parts[1]
                want_specific = True
        elif home_param and away_param:
            tgt_home, tgt_away = str(home_param), str(away_param)
            want_specific = True
        if want_specific and (df is None or df.empty) and not (df_all is None or df_all.empty):
            # Try robust match using abbreviations first
            cand = pd.DataFrame()
            try:
                if {'__home_abbr','__away_abbr'}.issubset(df_all.columns):
                    ht_ab = to_abbr_any(tgt_home)
                    at_ab = to_abbr_any(tgt_away)
                    cand = df_all[(df_all['__home_abbr'].astype(str) == ht_ab) & (df_all['__away_abbr'].astype(str) == at_ab)]
            except Exception:
                cand = pd.DataFrame()
            # Fallback to exact string match if abbr not available
            if cand is None or cand.empty:
                cand = df_all[(df_all["home_team"].astype(str) == str(tgt_home)) & (df_all["away_team"].astype(str) == str(tgt_away))]
            if not cand.empty:
                df = cand
                fallback_used = True
    except Exception:
        pass

    # Build games list (canonical labels). If filtered set is empty, use full set to ensure dropdown isn't empty.
    games = []
    try:
        gsrc = df
        if (gsrc is None or gsrc.empty) and (df_all is not None and not df_all.empty):
            gsrc = df_all
        if gsrc is not None and not gsrc.empty and {"home_team","away_team"}.issubset(gsrc.columns):
            gdf = gsrc[["home_team","away_team"]].dropna(how='any').drop_duplicates()
            for _, r in gdf.iterrows():
                ht = str(r.get("home_team") or "").strip()
                at = str(r.get("away_team") or "").strip()
                if not ht or not at:
                    continue
                games.append({"event": f"{at} @ {ht}", "home_team": ht, "away_team": at})
    except Exception:
        games = []

    # Default: return all games; only narrow when a specific event/home/away is requested
    # (Behavior changed from "first game unless all=1" to "all by default")
    # No action needed here; df already contains all rows unless upstream filtering set want_specific

    # Prepare output records
    records = []
    try:
        keep = [c for c in [
            "event","game_time","home_team","away_team",
            "market_key","market_name","period",
            "team_side","line","side","ev_units",
            "price_home","price_away","over_price","under_price","price",
            "edge_pts","is_alternate",
            # extended fields
            "threshold","range_low","range_high","range_type",
            "total_line","spread_line","total_side","winner","combo",
            "ht_result","ft_result","tie",
        ] if c in df.columns]
        if keep:
            for _, r in df[keep].iterrows():
                rec = {k: (None if (pd.isna(r.get(k)) if k in r else False) else r.get(k)) for k in keep}
                # Canonical event label
                try:
                    ht = r.get("home_team"); at = r.get("away_team")
                    if ht and at:
                        rec["event"] = f"{at} @ {ht}"
                except Exception:
                    pass
                # EV percent helper
                try:
                    if rec.get("ev_units") is not None:
                        rec["ev_pct"] = float(rec["ev_units"]) * 100.0
                except Exception:
                    rec["ev_pct"] = None
                records.append(rec)
    except Exception:
        records = []

    return jsonify({
        "season": season_i,
        "week": week_i,
        "games": games,
        "rows": len(records),
        "data": records,
        "source": str(edges_fp),
        "fallback_core_markets": fallback_used,
    })


if __name__ == "__main__":
    # Local dev: python app.py
    # Control debug and reloader via env to avoid churn in some environments (e.g., Windows + heavy deps)
    port = int(os.environ.get("PORT", 5050))
    debug = str(os.environ.get("FLASK_DEBUG", "0")).strip().lower() in {"1","true","yes","y"}
    use_reloader = str(os.environ.get("FLASK_USE_RELOADER", "0")).strip().lower() in {"1","true","yes","y"}
    # Threaded improves responsiveness; disable reloader by default (can be re-enabled with FLASK_USE_RELOADER=1)
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=use_reloader, threaded=True)
