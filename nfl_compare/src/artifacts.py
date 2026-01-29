from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


_ENV_DATA_DIR = os.environ.get("NFL_DATA_DIR")
DATA_DIR = Path(_ENV_DATA_DIR) if _ENV_DATA_DIR else (Path(__file__).resolve().parents[1] / "data")


@dataclass(frozen=True)
class ArtifactEntry:
    rel_path: str
    exists: bool
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    mtime_utc: Optional[str] = None
    rows: Optional[int] = None
    cols: Optional[List[str]] = None
    note: Optional[str] = None


@dataclass(frozen=True)
class WeekManifest:
    manifest_version: str
    created_utc: str
    season: int
    week: int
    data_dir: str
    artifacts: List[ArtifactEntry]
    checks: Dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_stat(path: Path) -> dict[str, Any]:
    try:
        st = path.stat()
        return {
            "size_bytes": int(st.st_size),
            "mtime_utc": datetime.utcfromtimestamp(st.st_mtime).isoformat(timespec="seconds") + "Z",
        }
    except Exception:
        return {}


def _safe_csv_shape_and_cols(path: Path) -> tuple[Optional[int], Optional[List[str]]]:
    try:
        # Local-only usage: ok to read full file.
        df = pd.read_csv(path)
        return int(df.shape[0]), list(df.columns)
    except Exception:
        return None, None


def _entry_for_file(data_dir: Path, path: Path, *, include_rows_cols: bool) -> ArtifactEntry:
    rel = str(path.relative_to(data_dir)).replace("\\", "/") if path.is_absolute() and data_dir in path.parents else str(path).replace("\\", "/")
    if not path.exists():
        return ArtifactEntry(rel_path=rel, exists=False)

    st = _safe_stat(path)
    rows = None
    cols = None
    if include_rows_cols and path.suffix.lower() == ".csv":
        rows, cols = _safe_csv_shape_and_cols(path)

    try:
        digest = sha256_file(path)
    except Exception:
        digest = None

    return ArtifactEntry(
        rel_path=rel,
        exists=True,
        sha256=digest,
        size_bytes=st.get("size_bytes"),
        mtime_utc=st.get("mtime_utc"),
        rows=rows,
        cols=cols,
    )


def default_week_manifest_path(season: int, week: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "manifests" / f"{int(season)}_wk{int(week)}.json"


def build_week_manifest(
    season: int,
    week: int,
    *,
    data_dir: Path = DATA_DIR,
    include_rows_cols: bool = True,
) -> WeekManifest:
    """Build a reproducibility manifest for a given (season, week).

    This is intended to run locally as part of the pipeline; production reads the JSON only.
    """
    s = int(season)
    w = int(week)

    backtest_dir = data_dir / "backtests" / f"{s}_wk{w}"

    expected_files: list[Path] = [
        data_dir / "games.csv",
        data_dir / "lines.csv",
        data_dir / "predictions.csv",
        data_dir / "predictions_week.csv",
        data_dir / "predictions_locked.csv",
        data_dir / "nfl_team_assets.json",
        data_dir / "pfr_drive_stats.csv",
        data_dir / f"player_props_{s}_wk{w}.csv",
        data_dir / f"player_props_vs_actuals_{s}_wk{w}.csv",
        # Player props markets + edges + ladders
        data_dir / f"oddsapi_player_props_{s}_wk{w}.csv",
        data_dir / f"bovada_player_props_{s}_wk{w}.csv",
        data_dir / f"edges_player_props_{s}_wk{w}.csv",
        data_dir / f"edges_player_props_{s}_wk{w}.json",
        data_dir / f"ladder_options_{s}_wk{w}.csv",
        data_dir / f"ladder_options_{s}_wk{w}.json",
        data_dir / f"player_props_market_source_{s}_wk{w}.json",
        # Game props markets + edges
        data_dir / f"bovada_game_props_{s}_wk{w}.csv",
        data_dir / f"edges_game_props_{s}_wk{w}.csv",
        data_dir / f"game_props_market_source_{s}_wk{w}.json",
        data_dir / "sigma_calibration.json",
        data_dir / "totals_calibration.json",
        data_dir / "prob_calibration.json",
        data_dir / "calibration_active.json",
        backtest_dir / "sim_probs.csv",
        backtest_dir / "sim_probs_scenarios.csv",
        backtest_dir / "sim_scenario_inputs.csv",
        backtest_dir / "sim_scenario_deltas.csv",
        backtest_dir / "sim_scenario_summary.csv",
        backtest_dir / "sim_quarters.csv",
        backtest_dir / "sim_drives.csv",
        backtest_dir / "sim_drives_scenarios.csv",
        backtest_dir / "sim_drives_scenarios_summary.csv",
        backtest_dir / "sim_scenarios.json",
        backtest_dir / "sim_scenarios_meta.json",
        backtest_dir / "player_props_scenarios.csv",
        backtest_dir / "player_props_scenarios_summary.csv",
        backtest_dir / "player_props_scenarios_meta.json",
        backtest_dir / "player_props_scenarios_accuracy.csv",
        backtest_dir / "player_props_scenarios_accuracy_by_position.csv",
        backtest_dir / "player_props_scenarios_coverage.csv",
        backtest_dir / "player_props_scenarios_top_errors.csv",
        backtest_dir / "player_props_scenarios_accuracy.md",
    ]

    artifacts = [_entry_for_file(data_dir, p, include_rows_cols=include_rows_cols) for p in expected_files]

    # Lightweight schema checks (local-only; shipped as precomputed JSON)
    checks: dict[str, Any] = {
        "backtest_dir": str(backtest_dir).replace("\\", "/"),
        "notes": [
            "Rows/cols are recorded for CSVs when include_rows_cols=true.",
            "This manifest is shipped-only in production; no computation at request time.",
        ],
    }

    required_cols: dict[str, list[str]] = {
        "games.csv": ["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"],
        f"backtests/{s}_wk{w}/sim_probs.csv": ["game_id", "pred_margin", "pred_total", "prob_home_win_mc"],
        f"backtests/{s}_wk{w}/sim_scenario_inputs.csv": ["scenario_id", "game_id"],
        f"backtests/{s}_wk{w}/sim_scenario_deltas.csv": ["scenario_id", "game_id"],
        f"backtests/{s}_wk{w}/sim_scenario_summary.csv": ["scenario_id"],
        f"backtests/{s}_wk{w}/sim_quarters.csv": ["game_id", "home_q1", "away_q1"],
        f"backtests/{s}_wk{w}/sim_drives.csv": ["game_id"],
        f"backtests/{s}_wk{w}/sim_drives_scenarios.csv": ["scenario_id", "game_id"],
        f"backtests/{s}_wk{w}/sim_drives_scenarios_summary.csv": ["scenario_id", "game_id"],
        f"backtests/{s}_wk{w}/player_props_scenarios.csv": ["scenario_id", "game_id", "team", "player"],
        f"backtests/{s}_wk{w}/player_props_scenarios_summary.csv": ["scenario_id", "game_id", "team"],
        f"backtests/{s}_wk{w}/player_props_scenarios_accuracy.csv": ["scenario_id", "n_rows"],
        f"backtests/{s}_wk{w}/player_props_scenarios_accuracy_by_position.csv": ["scenario_id", "position", "n_rows"],
        f"backtests/{s}_wk{w}/player_props_scenarios_coverage.csv": ["stat", "coverage_min_max"],
        f"backtests/{s}_wk{w}/player_props_scenarios_top_errors.csv": ["pick", "scenario_id", "player"],
        f"player_props_{s}_wk{w}.csv": ["game_id", "player"],
        f"edges_player_props_{s}_wk{w}.csv": ["player", "market", "line"],
        f"ladder_options_{s}_wk{w}.csv": ["player", "market", "line"],
        f"edges_game_props_{s}_wk{w}.csv": ["market"],
    }

    schema_results: dict[str, Any] = {}
    for rel, cols in required_cols.items():
        fp = data_dir / rel
        if not fp.exists():
            schema_results[rel] = {"exists": False, "ok": False, "missing_columns": cols}
            continue
        if fp.suffix.lower() != ".csv":
            schema_results[rel] = {"exists": True, "ok": True}
            continue
        try:
            df0 = pd.read_csv(fp, nrows=5)
            have = set(df0.columns)
            missing = [c for c in cols if c not in have]
            schema_results[rel] = {"exists": True, "ok": len(missing) == 0, "missing_columns": missing}
        except Exception as e:
            schema_results[rel] = {"exists": True, "ok": False, "error": str(e)}

    checks["schema"] = schema_results

    return WeekManifest(
        manifest_version="1.6",
        created_utc=_utc_now_iso(),
        season=s,
        week=w,
        data_dir=str(data_dir).replace("\\", "/"),
        artifacts=artifacts,
        checks=checks,
    )


def write_week_manifest(manifest: WeekManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = asdict(manifest)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def read_week_manifest(path: Path) -> dict[str, Any]:
    """Read a manifest file. Returns raw dict for API stability."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_manifest_for_week(season: int, week: int, *, data_dir: Path = DATA_DIR) -> Optional[Path]:
    fp = default_week_manifest_path(season, week, data_dir=data_dir)
    return fp if fp.exists() else None
