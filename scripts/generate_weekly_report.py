from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "nfl_compare" / "data"
REPORTS_DIR = BASE_DIR / "reports"


def load_csv(fp: Path) -> Optional[pd.DataFrame]:
    try:
        if fp.exists():
            df = pd.read_csv(fp)
            return df
    except Exception:
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate weekly backtest markdown report")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--in-dir", type=str, default=None, help="Input directory (default nfl_compare/data/backtests/<season>_wk<week>")
    ap.add_argument("--out", type=str, default=None, help="Output markdown file path")
    args = ap.parse_args()

    season = int(args.season)
    week = int(args.week)

    in_dir = Path(args.in_dir) if args.in_dir else (DATA_DIR / "backtests" / f"{season}_wk{week}")
    out_fp = Path(args.out) if args.out else (REPORTS_DIR / f"weekly_report_{season}_wk{week}.md")

    # Ensure reports dir
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    games_summary_fp = in_dir / "games_summary.csv"
    games_details_fp = in_dir / "games_details.csv"
    props_summary_fp = in_dir / "props_summary.csv"
    props_weekly_fp = in_dir / "props_weekly.csv"
    metrics_fp = in_dir / "metrics.json"

    games_summary = load_csv(games_summary_fp) or pd.DataFrame()
    props_summary = load_csv(props_summary_fp) or pd.DataFrame()

    # Load metrics
    metrics: dict = {}
    try:
        if metrics_fp.exists():
            metrics = json.loads(metrics_fp.read_text(encoding="utf-8"))
    except Exception:
        metrics = {}

    # Build markdown
    lines: list[str] = []
    lines.append(f"# Weekly Backtest Report — Season {season}, Week {week}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    if metrics:
        try:
            games_m = metrics.get("games") or {}
            props_m = metrics.get("props") or {}
            gm_str = ", ".join([f"{k}: {v:.3f}" for k, v in games_m.items() if isinstance(v, (int, float))])
            pm_str = ", ".join([f"{k}: {v:.3f}" for k, v in props_m.items() if isinstance(v, (int, float))])
            if gm_str:
                lines.append(f"- Games metrics: {gm_str}")
            if pm_str:
                lines.append(f"- Props metrics (mean of MAEs): {pm_str}")
        except Exception:
            pass
    else:
        lines.append("- No metrics.json found; see summary tables below.")
    lines.append("")

    # Games summary table
    if games_summary is not None and not games_summary.empty:
        lines.append("## Games Backtest Summary")
        lines.append("")
        try:
            # Format floats
            df = games_summary.copy()
            for c in ["mae_margin","mae_total","acc_home_win"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").round(3)
            lines.append(df.to_markdown(index=False))
        except Exception:
            lines.append("(failed to render games summary table)")
        lines.append("")
    else:
        lines.append("No games summary available.")
        lines.append("")

    # Props summary table
    if props_summary is not None and not props_summary.empty:
        lines.append("## Props Backtest Summary (MAE by position)")
        lines.append("")
        try:
            df = props_summary.copy()
            mae_cols = [c for c in df.columns if c.endswith("_MAE")]
            for c in mae_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
            lines.append(df.to_markdown(index=False))
        except Exception:
            lines.append("(failed to render props summary table)")
        lines.append("")
    else:
        lines.append("No props summary available.")
        lines.append("")

    # Files section
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Input dir: `{in_dir}`")
    if games_details_fp.exists():
        lines.append(f"- Games details: `{games_details_fp}`")
    if props_weekly_fp.exists():
        lines.append(f"- Props weekly: `{props_weekly_fp}`")
    if metrics_fp.exists():
        lines.append(f"- Metrics JSON: `{metrics_fp}`")
    lines.append("")

    # Calibration reliability (if present)
    try:
        prob_cal_fp = DATA_DIR / 'prob_calibration.json'
        if prob_cal_fp.exists():
            cal = json.loads(prob_cal_fp.read_text(encoding='utf-8'))
            lines.append("## Calibration Reliability (recent window)")
            lines.append("")
            def _render_curve(name: str, spec: dict | None):
                if not spec:
                    lines.append(f"- {name}: (no calibration)")
                    return
                xs = spec.get('xs') or []
                ys = spec.get('ys') or []
                if not xs or not ys or len(xs) != len(ys):
                    lines.append(f"- {name}: (invalid calibration)")
                    return
                df = pd.DataFrame({'p_raw_bin': xs, 'p_obs': ys})
                df['p_raw_bin'] = pd.to_numeric(df['p_raw_bin'], errors='coerce').round(3)
                df['p_obs'] = pd.to_numeric(df['p_obs'], errors='coerce').round(3)
                lines.append(f"### {name} calibration")
                lines.append("")
                lines.append(df.to_markdown(index=False))
                lines.append("")
            _render_curve('Moneyline (home win)', cal.get('moneyline'))
            _render_curve('ATS (home cover)', cal.get('ats'))
            _render_curve('Total (Over)', cal.get('total'))
            meta = cal.get('meta') or {}
            if meta:
                lines.append("Meta: " + ", ".join([f"{k}={v}" for k, v in meta.items()]))
                lines.append("")
    except Exception:
        pass

    # Calibration uplift summary (if evaluation artifacts present)
    try:
        eval_dir = DATA_DIR / 'backtests' / f'{season}_wk{week}'
        eval_fp = eval_dir / 'calibration_eval.json'
        if eval_fp.exists():
            ev = json.loads(eval_fp.read_text(encoding='utf-8'))
            lines.append("## Calibration Uplift Summary")
            lines.append("")
            for key, name in [('moneyline','Moneyline (home win)'), ('ats','ATS (home cover)'), ('total','Total (Over)')]:
                spec = ev.get(key) or {}
                br_raw = spec.get('brier_raw')
                br_cal = spec.get('brier_cal')
                delta = None
                try:
                    if br_raw is not None and br_cal is not None:
                        delta = float(br_raw) - float(br_cal)
                except Exception:
                    delta = None
                lines.append(f"- {name}: Brier raw={br_raw:.5f} cal={br_cal:.5f} Δ={delta:.5f}" if (isinstance(br_raw,(int,float)) and isinstance(br_cal,(int,float))) else f"- {name}: (insufficient data)")
            lines.append("")
    except Exception:
        pass

    out_fp.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote weekly report -> {out_fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
