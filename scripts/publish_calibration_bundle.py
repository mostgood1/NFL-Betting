"""Publish a versioned calibration bundle for provenance + deterministic pinning.

This script snapshots the *current* calibration artifacts:
  - nfl_compare/data/sigma_calibration.json
  - nfl_compare/data/totals_calibration.json
  - nfl_compare/data/prob_calibration.json

into a versioned folder under:
  nfl_compare/data/calibration_bundles/

and writes/updates a small pointer file:
  nfl_compare/data/calibration_active.json

That pointer is what manifests should capture; it records which bundle was active
(and hashes) when sims/artifacts were generated.

Usage:
  python scripts/publish_calibration_bundle.py --season 2025 --week 10 --sigma-lookback 4 --prob-lookback 6 --totals-weeks 4

Notes:
- This is shipped-only: production should read calibration JSONs; no fitting here.
- Safe: if some calibration files are missing, it still writes calibration_active.json
  (with exists=false entries), so manifests remain informative.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATA_DIR = ROOT / "nfl_compare" / "data"


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _sha256_file(path: Path) -> Optional[str]:
    try:
        import hashlib

        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                b = f.read(1024 * 1024)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class CalFile:
    name: str
    src: Path


def _copy_if_exists(src: Path, dst: Path) -> bool:
    try:
        if not src.exists():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Snapshot current calibration files into a versioned bundle")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--sigma-lookback", type=int, default=4)
    ap.add_argument("--prob-lookback", type=int, default=6)
    ap.add_argument("--totals-weeks", type=int, default=4)
    ap.add_argument("--out-root", type=str, default=str(DATA_DIR / "calibration_bundles"))
    ap.add_argument("--tag", type=str, default=None, help="Optional tag for bundle folder name")
    args = ap.parse_args()

    season = int(args.season)
    week = int(args.week)

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    tag = str(args.tag).strip() if args.tag else None
    folder = f"{season}_wk{week}_{stamp}" + (f"_{tag}" if tag else "")
    out_root = Path(args.out_root)
    out_dir = out_root / folder

    cal_files = [
        CalFile("sigma", DATA_DIR / "sigma_calibration.json"),
        CalFile("totals", DATA_DIR / "totals_calibration.json"),
        CalFile("prob", DATA_DIR / "prob_calibration.json"),
    ]

    active: Dict[str, Any] = {
        "created_utc": _utc_now(),
        "season": season,
        "week": week,
        "bundle_dir": str(out_dir.relative_to(DATA_DIR)).replace("\\", "/") if (DATA_DIR in out_dir.parents) else str(out_dir).replace("\\", "/"),
        "params": {
            "sigma_lookback": int(args.sigma_lookback),
            "prob_lookback": int(args.prob_lookback),
            "totals_weeks": int(args.totals_weeks),
        },
        "files": {},
    }

    # Copy files into bundle and compute hashes
    for cf in cal_files:
        dst = out_dir / cf.src.name
        ok = _copy_if_exists(cf.src, dst)
        entry: Dict[str, Any] = {
            "src": str(cf.src.relative_to(DATA_DIR)).replace("\\", "/") if (DATA_DIR in cf.src.parents) else str(cf.src).replace("\\", "/"),
            "dst": str(dst.relative_to(DATA_DIR)).replace("\\", "/") if (DATA_DIR in dst.parents) else str(dst).replace("\\", "/"),
            "exists": bool(ok),
            "sha256": _sha256_file(dst) if ok else None,
        }
        active["files"][cf.name] = entry

    # Write bundle meta.json
    bundle_meta = {
        "created_utc": active["created_utc"],
        "season": season,
        "week": week,
        "params": active["params"],
        "files": active["files"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(out_dir / "meta.json", json.dumps(bundle_meta, indent=2) + "\n")

    # Write/overwrite calibration_active.json
    _atomic_write_text(DATA_DIR / "calibration_active.json", json.dumps(active, indent=2) + "\n")

    print(json.dumps({"ok": True, "bundle": str(out_dir).replace("\\", "/"), "active": str((DATA_DIR / 'calibration_active.json')).replace("\\", "/")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
