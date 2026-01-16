from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd

# Data directory within nfl_compare
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def _norm(s: object) -> str:
    return str(s or "").strip().lower()


def _map_pos(p: object) -> str:
    p = str(p or "").upper()
    if p in {"HB", "FB"}:
        return "RB"
    if p.startswith("WR"):
        return "WR"
    if p.startswith("TE"):
        return "TE"
    if p.startswith("RB"):
        return "RB"
    if p.startswith("QB"):
        return "QB"
    return p


# ESPN team abbreviations used in depth chart URLs
ESPN_TEAM_ABBR = {
    "Arizona Cardinals": "ari",
    "Atlanta Falcons": "atl",
    "Baltimore Ravens": "bal",
    "Buffalo Bills": "buf",
    "Carolina Panthers": "car",
    "Chicago Bears": "chi",
    "Cincinnati Bengals": "cin",
    "Cleveland Browns": "cle",
    "Dallas Cowboys": "dal",
    "Denver Broncos": "den",
    "Detroit Lions": "det",
    "Green Bay Packers": "gb",
    "Houston Texans": "hou",
    "Indianapolis Colts": "ind",
    "Jacksonville Jaguars": "jax",
    "Kansas City Chiefs": "kc",
    "Las Vegas Raiders": "lv",
    "Los Angeles Chargers": "lac",
    "Los Angeles Rams": "lar",
    "Miami Dolphins": "mia",
    "Minnesota Vikings": "min",
    "New England Patriots": "ne",
    "New Orleans Saints": "no",
    "New York Giants": "nyg",
    "New York Jets": "nyj",
    "Philadelphia Eagles": "phi",
    "Pittsburgh Steelers": "pit",
    "San Francisco 49ers": "sf",
    "Seattle Seahawks": "sea",
    "Tampa Bay Buccaneers": "tb",
    "Tennessee Titans": "ten",
    "Washington Commanders": "wsh",
}


def _extract_status_and_clean(txt: str) -> tuple[str, str]:
    """Return (clean_name, status_token) from a name cell such as 'T.J. Hockenson Q'.
    Status tokens recognized: Q (Questionable), D (Doubtful), O (Out), IR, PUP, NFI, SUSP, DNP.
    """
    raw = (txt or "").strip()
    # Capture trailing status token(s)
    m = re.search(r"\s+(Q|D|O|IR|PUP|NFI|SUSP|DNP)$", raw, flags=re.IGNORECASE)
    status = (m.group(1).upper() if m else "")
    # Clean markers and asterisks from the name portion
    name = raw
    if m:
        name = raw[: m.start()].strip()
    name = name.replace("*", "").strip()
    return name, status


def _status_to_active(status: str) -> bool:
    s = (status or "").upper()
    # Treat clear out designations as inactive; Q/D remain active for projections
    if s in {"O", "IR", "PUP", "NFI", "SUSP", "DNP"}:
        return False
    return True


def _norm_col_name(c: object) -> str:
    return str(c or "").strip().lower()


def _pick_offense_table(tables: list[pd.DataFrame]) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Pick the most likely *offense* depth chart table from pd.read_html output.

    ESPN pages typically contain multiple depth chart tables (offense/defense/special teams)
    with similar tier column names. We score candidate tables based on whether their
    position labels contain QB/RB/WR/TE and prefer those.

    Returns (table, position_col_name). position_col_name may be None if position labels
    appear in the index.
    """
    best: tuple[int, Optional[pd.DataFrame], Optional[str]] = (-10**9, None, None)
    tier_norms = {"starter", "2nd", "3rd", "4th"}
    offense_tokens = {"QB", "RB", "HB", "FB", "WR", "TE"}
    defense_tokens = {
        "LDE", "RDE", "DE", "DT", "NT", "DL",
        "MLB", "ILB", "OLB", "LB",
        "CB", "S", "SS", "FS",
    }

    for t in tables:
        if not isinstance(t, pd.DataFrame) or t.empty:
            continue

        cols = list(t.columns)
        norm_cols = {_norm_col_name(c): c for c in cols}
        if "starter" not in norm_cols or "2nd" not in norm_cols:
            continue

        # Identify a likely position-label column (if any)
        pos_col = None
        non_tier_cols = [c for c in cols if _norm_col_name(c) not in tier_norms]
        if non_tier_cols:
            pos_col = non_tier_cols[0]

        if pos_col is not None:
            pos_vals = set(
                str(v).strip().upper()
                for v in t[pos_col].head(30).tolist()
                if pd.notna(v) and str(v).strip()
            )
        else:
            pos_vals = set(
                str(v).strip().upper()
                for v in list(t.index)[:30]
                if pd.notna(v) and str(v).strip()
            )

        score = 0
        if "QB" in pos_vals:
            score += 25
        if pos_vals & {"RB", "HB", "FB"}:
            score += 10
        if any(v == "WR" or v.startswith("WR") for v in pos_vals):
            score += 10
        if any(v == "TE" or v.startswith("TE") for v in pos_vals):
            score += 10
        if pos_vals & offense_tokens:
            score += 2
        if pos_vals & defense_tokens:
            score -= 10

        if score > best[0]:
            best = (score, t, pos_col)

    # Accept a non-negative score as a plausible offense table. Some ESPN pages
    # don't expose clean position tokens in the parsed HTML, resulting in score==0.
    # We still prefer that over clearly defensive tables (negative score).
    if best[1] is None or best[0] < 0:
        return None, None
    return best[1].copy(), best[2]


def _scrape_espn_depth_for_team(team_name: str) -> dict[str, list[str]]:
    """Return pos -> ordered list scraped from ESPN depth chart page for a team."""
    import requests
    from bs4 import BeautifulSoup

    abbr = ESPN_TEAM_ABBR.get(team_name)
    if not abbr:
        return {}
    url = f"https://www.espn.com/nfl/team/depth/_/name/{abbr}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception:
        return {}

    soup = BeautifulSoup(resp.text, "lxml")

    # Parse formation tokens if present (e.g., "3WR 1TE 1RB")
    formation_text = ""
    try:
        txt_nodes = soup.find_all(string=True)
        cand = [t.strip() for t in txt_nodes if t and "WR" in t and ("TE" in t or "RB" in t)]
        cand = sorted(cand, key=lambda s: len(s))
        if cand:
            formation_text = cand[0]
    except Exception:
        formation_text = ""
    wr_count = 3; te_count = 1; rb_count = 1
    if formation_text:
        wr_m = re.search(r"(\d+)\s*WR", formation_text, flags=re.IGNORECASE)
        te_m = re.search(r"(\d+)\s*TE", formation_text, flags=re.IGNORECASE)
        rb_m = re.search(r"(\d+)\s*RB", formation_text, flags=re.IGNORECASE)
        if wr_m:
            wr_count = max(1, int(wr_m.group(1)))
        if te_m:
            te_count = max(1, int(te_m.group(1)))
        if rb_m:
            rb_count = max(1, int(rb_m.group(1)))

    # Extract the offense tiers table via pandas
    try:
        # Wrap literal HTML string in StringIO to avoid pandas deprecation
        tables = pd.read_html(StringIO(resp.text))
    except Exception:
        return {}

    df_raw, pos_col = _pick_offense_table(tables)
    if df_raw is None:
        return {}

    # Normalize/locate tier columns and optional position column
    norm_cols = {_norm_col_name(c): c for c in df_raw.columns}
    starter_c = norm_cols.get("starter")
    second_c = norm_cols.get("2nd")
    third_c = norm_cols.get("3rd")
    fourth_c = norm_cols.get("4th")
    if starter_c is None or second_c is None:
        return {}

    # Position labels may live in a dedicated column or in the index.
    if pos_col is not None and pos_col in df_raw.columns:
        pos_series = df_raw[pos_col].astype(str).str.strip().str.upper()
    else:
        pos_series = pd.Series(list(df_raw.index), index=df_raw.index).astype(str).str.strip().str.upper()

    df = pd.DataFrame({
        "pos": pos_series,
        "Starter": df_raw[starter_c],
        "2nd": df_raw[second_c],
        "3rd": (df_raw[third_c] if third_c is not None else ""),
        "4th": (df_raw[fourth_c] if fourth_c is not None else ""),
    })
    for c in ["Starter", "2nd", "3rd", "4th"]:
        df[c] = df[c].astype(str).str.strip().replace({"-": "", "nan": ""})

    def _is_pos(p: str, want: str) -> bool:
        p = (p or "").strip().upper()
        want = (want or "").strip().upper()
        if want == "WR":
            return p == "WR" or p.startswith("WR")
        if want == "TE":
            return p == "TE" or p.startswith("TE")
        if want == "RB":
            return p in {"RB", "HB", "FB"} or p.startswith("RB")
        return p == want or p.startswith(want)

    pos_lists: dict[str, list[dict]] = {"QB": [], "RB": [], "WR": [], "TE": []}
    tiers = ["Starter", "2nd", "3rd", "4th"]

    # QB: single row
    qb_rows = df[df["pos"].map(lambda p: _is_pos(p, "QB"))]
    if not qb_rows.empty:
        row = qb_rows.iloc[0]
        for tier in tiers:
            nm, st = _extract_status_and_clean(row[tier])
            if nm and all(nm != x.get("player") for x in pos_lists["QB"]):
                pos_lists["QB"].append({"player": nm, "status": st, "active": _status_to_active(st)})

    # RB: take up to rb_count matching rows (often just one)
    rb_rows = df[df["pos"].map(lambda p: _is_pos(p, "RB"))].head(max(1, rb_count))
    for _, row in rb_rows.iterrows():
        for tier in tiers:
            nm, st = _extract_status_and_clean(row[tier])
            if nm and all(nm != x.get("player") for x in pos_lists["RB"]):
                pos_lists["RB"].append({"player": nm, "status": st, "active": _status_to_active(st)})

    # WR: aggregate across WR rows by tier to represent multi-WR formations
    wr_rows = df[df["pos"].map(lambda p: _is_pos(p, "WR"))].head(max(1, wr_count))
    if not wr_rows.empty:
        for tier in tiers:
            for _, row in wr_rows.iterrows():
                nm, st = _extract_status_and_clean(row[tier])
                if nm and all(nm != x.get("player") for x in pos_lists["WR"]):
                    pos_lists["WR"].append({"player": nm, "status": st, "active": _status_to_active(st)})

    # TE: take up to te_count matching rows (often just one)
    te_rows = df[df["pos"].map(lambda p: _is_pos(p, "TE"))].head(max(1, te_count))
    for _, row in te_rows.iterrows():
        for tier in tiers:
            nm, st = _extract_status_and_clean(row[tier])
            if nm and all(nm != x.get("player") for x in pos_lists["TE"]):
                pos_lists["TE"].append({"player": nm, "status": st, "active": _status_to_active(st)})

    # Fallback: if we couldn't identify position-labeled rows, revert to the prior
    # row-order parsing against the top "skill" rows.
    if not any(pos_lists.values()):
        df_skill = df[["Starter", "2nd", "3rd", "4th"]].copy().reset_index(drop=True)
        skill_rows = min(1 + rb_count + wr_count + te_count, len(df_skill))
        df_skill = df_skill.iloc[:skill_rows].copy()

        # QB row
        if skill_rows >= 1:
            for tier in tiers:
                nm, st = _extract_status_and_clean(df_skill.loc[0, tier])
                if nm and all(nm != x.get("player") for x in pos_lists["QB"]):
                    pos_lists["QB"].append({"player": nm, "status": st, "active": _status_to_active(st)})

        # RB rows
        idx = 1
        for _ in range(rb_count):
            if idx >= skill_rows:
                break
            for tier in tiers:
                nm, st = _extract_status_and_clean(df_skill.loc[idx, tier])
                if nm and all(nm != x.get("player") for x in pos_lists["RB"]):
                    pos_lists["RB"].append({"player": nm, "status": st, "active": _status_to_active(st)})
            idx += 1

        # WR rows aggregated by tier
        wr_rows = []
        for _ in range(wr_count):
            if idx >= skill_rows:
                break
            wr_rows.append(df_skill.loc[idx, tiers].tolist())
            idx += 1
        if wr_rows:
            for tier_idx in range(len(tiers)):
                for r in wr_rows:
                    nm, st = _extract_status_and_clean((r[tier_idx] or "").strip())
                    if nm and all(nm != x.get("player") for x in pos_lists["WR"]):
                        pos_lists["WR"].append({"player": nm, "status": st, "active": _status_to_active(st)})

        # TE rows
        for _ in range(te_count):
            if idx >= skill_rows:
                break
            for tier in tiers:
                nm, st = _extract_status_and_clean(df_skill.loc[idx, tier])
                if nm and all(nm != x.get("player") for x in pos_lists["TE"]):
                    pos_lists["TE"].append({"player": nm, "status": st, "active": _status_to_active(st)})
            idx += 1

    return {k: v for k, v in pos_lists.items() if v}


def build_depth_chart_from_espn(season: int, week: int, teams: Optional[list[str]] = None) -> pd.DataFrame:
    rows = []
    team_list = list(teams) if teams else list(ESPN_TEAM_ABBR.keys())
    for team in team_list:
        pos_lists = _scrape_espn_depth_for_team(team)
        for pos, players in pos_lists.items():
            depth_size = len(players)
            for i, row in enumerate(players, start=1):
                rows.append({
                    "season": season,
                    "week": week,
                    "team": team,
                    "position": pos,
                    "player": row.get("player"),
                    "depth_rank": i,
                    "depth_size": depth_size,
                    "status": row.get("status", ""),
                    "active": bool(row.get("active", True)),
                })
    return pd.DataFrame(rows)


def build_depth_chart(season: int, week: int, source: str = "espn", teams: Optional[list[str]] = None) -> pd.DataFrame:
    """Build a weekly depth chart. Default source uses ESPN web page parsing.
    Returns columns: season, week, team, position, player, depth_rank, depth_size, status, active.
    """
    if source == "espn":
        df = build_depth_chart_from_espn(season, week, teams=teams)
    else:
        df = build_depth_chart_from_espn(season, week, teams=teams)
    return df


def save_depth_chart(season: int, week: int, source: str = "espn", teams: Optional[list[str]] = None) -> Path:
    df = build_depth_chart(season, week, source=source, teams=teams)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"depth_chart_{season}_wk{week}.csv"

    # If updating only a subset of teams, patch them into the existing CSV if present.
    # Never delete a team's prior rows unless we successfully scraped replacement rows.
    if teams and out.exists():
        try:
            prev = pd.read_csv(out)
            keep_prev = prev.copy()
            for t in teams:
                if (df["team"] == t).any():
                    keep_prev = keep_prev[keep_prev["team"] != t]
            df = pd.concat([keep_prev, df], ignore_index=True)
        except Exception:
            # Fall back to writing just the new scrape.
            pass

    df.to_csv(out, index=False)
    return out


def load_depth_chart_csv(season: int, week: int) -> pd.DataFrame:
    fp = DATA_DIR / f"depth_chart_{season}_wk{week}.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(fp)
        return df
    except Exception:
        return pd.DataFrame()
