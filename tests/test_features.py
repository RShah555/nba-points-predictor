"""Checks that no feature ever uses the game it is predicting."""
import pandas as pd

from model import FEATURES, add_player_features, build_training_frame, normalize


def fake_logs(points):
    n = len(points)
    return pd.DataFrame({
        'SEASON_ID': ['22025'] * n,
        'PLAYER_ID': [1] * n,
        'GAME_ID': [f'g{i}' for i in range(n)],
        'GAME_DATE': pd.date_range('2025-10-22', periods=n, freq='2D').strftime('%Y-%m-%d'),
        'MATCHUP': ['BOS vs. NYK' if i % 2 == 0 else 'BOS @ NYK' for i in range(n)],
        'PTS': points, 'MIN': [30] * n, 'FGA': [15] * n, 'FG3A': [5] * n, 'FTA': [4] * n,
        'AST': [5] * n, 'REB': [6] * n, 'TOV': [2] * n,
    })


def test_rolling_features_only_use_past_games():
    df = add_player_features(normalize(fake_logs([10, 20, 30, 40, 50, 60])))
    # Before game 4 (40 points), the last three games scored 10, 20 and 30
    assert df.loc[3, 'PTS_roll_3'] == 20
    assert df.loc[3, 'SEASON_AVG_PTS'] == 20
    # The first game has no history at all
    assert pd.isna(df.loc[0, 'PTS_roll_3'])


def test_changing_a_game_never_changes_its_own_features():
    a = add_player_features(normalize(fake_logs([10, 20, 30, 40, 50, 60])))
    b = add_player_features(normalize(fake_logs([10, 20, 30, 99, 50, 60])))
    rolling = [c for c in FEATURES if 'roll' in c or 'SEASON_AVG' in c]
    pd.testing.assert_frame_equal(a.loc[:3, rolling], b.loc[:3, rolling])


def test_home_and_rest_days():
    df = add_player_features(normalize(fake_logs([10] * 4)))
    assert df['HOME'].tolist() == [1, 0, 1, 0]
    assert df.loc[1, 'REST_DAYS'] == 2


def test_training_frame_has_every_feature():
    df, _ = build_training_frame(fake_logs(list(range(10, 30))))
    assert set(FEATURES) <= set(df.columns)
