from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("NFL_DATA_DIR", str(ROOT / "nfl_compare" / "data"))).resolve()
CACHE_DIR = DATA_DIR / "external" / "nfl_data_py"


def _atomic_write_csv_gz(df: pd.DataFrame, out_fp: Path) -> None:
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_fp.with_suffix(out_fp.suffix + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(out_fp)


def _atomic_write_parquet(df: pd.DataFrame, out_fp: Path) -> bool:
    """Return True if parquet write succeeded."""
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_fp.with_suffix(out_fp.suffix + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
        tmp.replace(out_fp)
        return True
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def _read_cached_table(stem: str) -> Optional[pd.DataFrame]:
    pq = CACHE_DIR / f"{stem}.parquet"
    gz = CACHE_DIR / f"{stem}.csv.gz"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    if gz.exists():
        try:
            return pd.read_csv(gz)
        except Exception:
            pass
    return None


def _write_cached_table(df: pd.DataFrame, stem: str) -> None:
    pq = CACHE_DIR / f"{stem}.parquet"
    if _atomic_write_parquet(df, pq):
        return
    _atomic_write_csv_gz(df, CACHE_DIR / f"{stem}.csv.gz")


def get_seasonal_rosters(season: int, *, refresh: bool = False, timeout_sec: float = 30.0) -> pd.DataFrame:
    """Load nfl_data_py seasonal rosters for a season, cached locally.

    Cache location: nfl_compare/data/external/nfl_data_py/seasonal_rosters_{season}.(parquet|csv.gz)

    If cache is missing (or refresh=True), fetches using nfl_data_py and writes the cache.
    On failure, returns empty DataFrame.
    """
    season_i = int(season)
    stem = f"seasonal_rosters_{season_i}"

    if not refresh:
        cached = _read_cached_table(stem)
        if cached is not None:
            return cached

    try:
        import nfl_data_py as nfl  # type: ignore

        prev_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(float(timeout_sec))
        try:
            df = nfl.import_seasonal_rosters([season_i])
        finally:
            socket.setdefaulttimeout(prev_timeout)
        if df is None:
            return pd.DataFrame()
        df = df.copy()
        _write_cached_table(df, stem)
        return df
    except Exception:
        return pd.DataFrame()


def get_weekly_rosters(season: int, *, refresh: bool = False, timeout_sec: float = 30.0) -> pd.DataFrame:
    """Load nfl_data_py weekly rosters for a season, cached locally.

    Cache location: nfl_compare/data/external/nfl_data_py/weekly_rosters_{season}.(parquet|csv.gz)

    If cache is missing (or refresh=True), fetches using nfl_data_py and writes the cache.
    On failure, returns empty DataFrame.

    Notes:
    - nfl_data_py.import_weekly_rosters sometimes touches remote CSV endpoints; we apply a socket timeout.
    - The returned dataframe is the full season table; filter by week upstream.
    """
    season_i = int(season)
    stem = f"weekly_rosters_{season_i}"

    if not refresh:
        cached = _read_cached_table(stem)
        if cached is not None:
            return cached

    try:
        import nfl_data_py as nfl  # type: ignore

        prev_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(float(timeout_sec))
        try:
            df = nfl.import_weekly_rosters([season_i])
        finally:
            socket.setdefaulttimeout(prev_timeout)
        if df is None:
            return pd.DataFrame()
        df = df.copy()
        _write_cached_table(df, stem)
        return df
    except Exception:
        return pd.DataFrame()
