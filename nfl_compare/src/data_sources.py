import pandas as pd
from pathlib import Path
from .schemas import GameRow, TeamStatRow, LineRow

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# CSV readers; swap out to real APIs when ready

def load_games() -> pd.DataFrame:
    fp = DATA_DIR / "games.csv"
    if not fp.exists():
        return pd.DataFrame(columns=GameRow.model_fields.keys())
    df = pd.read_csv(fp)
    return df

def load_team_stats() -> pd.DataFrame:
    fp = DATA_DIR / "team_stats.csv"
    if not fp.exists():
        return pd.DataFrame(columns=TeamStatRow.model_fields.keys())
    df = pd.read_csv(fp)
    return df

def load_lines() -> pd.DataFrame:
    fp = DATA_DIR / "lines.csv"
    if not fp.exists():
        return pd.DataFrame(columns=LineRow.model_fields.keys())
    df = pd.read_csv(fp)
    return df

def load_predictions() -> pd.DataFrame:
    """Lightweight reader for predictions.csv used as a fallback source in pipelines.
    Returns an empty DataFrame if the file is missing.
    """
    fp = DATA_DIR / "predictions.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()
    return df

