"""Train and evaluate the next game points model.

Pulls every player game log for each season with one LeagueGameLog call,
trains on the earlier seasons and tests on a season the model never saw.

    python train_model.py
"""
import json
import os
import time

import joblib
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from sklearn.ensemble import HistGradientBoostingRegressor

from model import FEATURES, build_training_frame, score

TRAIN_SEASONS = ['2023-24', '2024-25']
TEST_SEASON = '2025-26'
DATA_DIR = 'data'
MODEL_PATH = os.path.join('models', 'points_model.joblib')


def load_season(season):
    path = os.path.join(DATA_DIR, f'player_logs_{season}.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f'Downloading {season} player game logs...')
    df = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='P', timeout=120).get_data_frames()[0]
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    time.sleep(1)  # be polite to stats.nba.com
    return df


def main():
    raw = pd.concat([load_season(s) for s in TRAIN_SEASONS + [TEST_SEASON]], ignore_index=True)
    df, latest_defense = build_training_frame(raw)

    test_id = '2' + TEST_SEASON[:4]  # stats.nba.com SEASON_ID, e.g. 22025 for 2025-26
    test_mask = df['SEASON_ID'].astype(str) == test_id
    train, test = df[~test_mask], df[test_mask]
    print(f'Training on {len(train):,} player games, testing on {len(test):,} from {TEST_SEASON}')

    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=42,
    )
    model.fit(train[FEATURES], train['PTS'])

    results = {
        'Season average so far': score(test['PTS'], test['SEASON_AVG_PTS']),
        'Last 5 game average': score(test['PTS'], test['PTS_roll_5']),
        'Gradient boosting model': score(test['PTS'], model.predict(test[FEATURES])),
    }
    best_baseline = min(results['Season average so far']['MAE'], results['Last 5 game average']['MAE'])
    model_mae = results['Gradient boosting model']['MAE']
    improvement = round(100 * (best_baseline - model_mae) / best_baseline, 1)

    print(f"\n{'Method':<26}{'MAE':>7}{'RMSE':>8}")
    for name, m in results.items():
        print(f"{name:<26}{m['MAE']:>7}{m['RMSE']:>8}")
    print(f'\nModel beats the best baseline by {improvement}% on MAE')

    metrics = {
        'train_seasons': TRAIN_SEASONS, 'test_season': TEST_SEASON,
        'train_rows': len(train), 'test_rows': len(test),
        'results': results, 'improvement_pct': improvement,
    }
    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': model, 'features': FEATURES, 'metrics': metrics,
                 'defense': latest_defense}, MODEL_PATH)
    with open(os.path.join('models', 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Saved {MODEL_PATH} and models/metrics.json')
    write_readme_results(metrics)


def write_readme_results(metrics):
    """Replace the results table in README.md with this run's numbers."""
    start, end = '<!-- RESULTS:START -->', '<!-- RESULTS:END -->'
    if not os.path.exists('README.md'):
        return
    with open('README.md') as f:
        readme = f.read()
    if start not in readme or end not in readme:
        return
    lines = [
        f"Tested on {metrics['test_rows']:,} player games from the {metrics['test_season']} season, "
        f"trained on {metrics['train_rows']:,} games from {' and '.join(metrics['train_seasons'])}.",
        '',
        '| Method | MAE (points) | RMSE (points) |',
        '| --- | --- | --- |',
    ]
    for name, m in metrics['results'].items():
        lines.append(f"| {name} | {m['MAE']} | {m['RMSE']} |")
    lines += ['', f"The model's average miss is {metrics['improvement_pct']}% smaller than the best baseline."]
    before, rest = readme.split(start, 1)
    after = rest.split(end, 1)[1]
    with open('README.md', 'w') as f:
        f.write(before + start + '\n' + '\n'.join(lines) + '\n' + end + after)
    print('Updated the results table in README.md')


if __name__ == '__main__':
    main()
