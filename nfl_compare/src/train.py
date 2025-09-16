import pandas as pd
from pathlib import Path
from joblib import dump
from .data_sources import load_games, load_team_stats, load_lines
from .features import merge_features
from .models import train_models

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    games = load_games()
    team_stats = load_team_stats()
    lines = load_lines()

    if games.empty:
        print('No games.csv found. Add historical data to train models.')
        return

    df = merge_features(games, team_stats, lines)
    # Filter rows with targets available
    df_train = df.dropna(subset=['home_score','away_score'])

    models = train_models(df_train)
    dump(models, MODELS_DIR / 'nfl_models.joblib')
    print('Models trained and saved to models/nfl_models.joblib')


if __name__ == '__main__':
    main()
