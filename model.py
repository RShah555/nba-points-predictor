"""Feature engineering and evaluation for the next game points model.

Every feature for a game is built only from games played before it
(shift(1) before any rolling window), so the model never sees the
answer it is trying to predict.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

STATS = ['PTS', 'MIN', 'FGA', 'FG3A', 'FTA', 'AST', 'REB', 'TOV']
WINDOWS = [3, 5, 10]
MIN_PRIOR_GAMES = 5  # skip a player's first few games, there is no history yet

FEATURES = (
    [f'{s}_roll_{w}' for s in STATS for w in WINDOWS]
    + ['SEASON_AVG_PTS', 'SEASON_AVG_MIN', 'GAMES_PLAYED', 'HOME', 'REST_DAYS', 'BACK_TO_BACK', 'OPP_PTS_ALLOWED']
)


def normalize(df):
    """Make PlayerGameLog and LeagueGameLog frames look the same."""
    df = df.rename(columns={'Player_ID': 'PLAYER_ID', 'Game_ID': 'GAME_ID'}).copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='mixed')
    df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
    df['HOME'] = df['MATCHUP'].str.contains('vs.', regex=False).astype(int)
    df['TEAM'] = df['MATCHUP'].str.split(' ').str[0]
    df['OPP'] = df['MATCHUP'].str.split(' ').str[-1]
    return df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)


def team_defense(df):
    """Average points each team allowed in its games before a given date.

    Built from the player logs themselves: points a team allowed in a game
    are the points the other team's players scored against it.
    """
    scored = df.groupby(['SEASON_ID', 'GAME_ID', 'GAME_DATE', 'OPP'], as_index=False)['PTS'].sum()
    allowed = scored.rename(columns={'OPP': 'DEF_TEAM', 'PTS': 'ALLOWED'}).sort_values(['DEF_TEAM', 'GAME_DATE'])
    allowed['OPP_PTS_ALLOWED'] = (
        allowed.groupby(['DEF_TEAM', 'SEASON_ID'])['ALLOWED'].transform(lambda s: s.shift(1).expanding().mean())
    )
    latest = allowed.groupby('DEF_TEAM')['ALLOWED'].apply(lambda s: float(s.tail(20).mean())).to_dict()
    return allowed[['DEF_TEAM', 'GAME_DATE', 'OPP_PTS_ALLOWED']], latest


def add_player_features(df):
    g = df.groupby('PLAYER_ID')
    for stat in STATS:
        for w in WINDOWS:
            df[f'{stat}_roll_{w}'] = g[stat].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
    gs = df.groupby(['PLAYER_ID', 'SEASON_ID'])
    df['SEASON_AVG_PTS'] = gs['PTS'].transform(lambda s: s.shift(1).expanding().mean())
    df['SEASON_AVG_MIN'] = gs['MIN'].transform(lambda s: s.shift(1).expanding().mean())
    df['GAMES_PLAYED'] = gs.cumcount()
    rest = g['GAME_DATE'].diff().dt.days
    df['REST_DAYS'] = rest.clip(upper=7).fillna(3)
    df['BACK_TO_BACK'] = (df['REST_DAYS'] == 1).astype(int)
    return df


def build_training_frame(raw):
    df = add_player_features(normalize(raw))
    defense, latest = team_defense(df)
    df = df.merge(defense, left_on=['OPP', 'GAME_DATE'], right_on=['DEF_TEAM', 'GAME_DATE'],
                  how='left').drop(columns=['DEF_TEAM'])
    df = df[df['GAMES_PLAYED'] >= MIN_PRIOR_GAMES]
    return df.reset_index(drop=True), latest


def next_game_features(player_log, home, rest_days, opp_pts_allowed):
    """Feature row for a player's next game, using every game played so far."""
    df = normalize(player_log)
    row = {}
    for stat in STATS:
        for w in WINDOWS:
            row[f'{stat}_roll_{w}'] = df[stat].tail(w).mean()
    season = df[df['SEASON_ID'] == df['SEASON_ID'].iloc[-1]]
    row['SEASON_AVG_PTS'] = season['PTS'].mean()
    row['SEASON_AVG_MIN'] = season['MIN'].mean()
    row['GAMES_PLAYED'] = len(season)
    row['HOME'] = int(home)
    row['REST_DAYS'] = min(rest_days, 7)
    row['BACK_TO_BACK'] = int(rest_days == 1)
    row['OPP_PTS_ALLOWED'] = opp_pts_allowed
    return pd.DataFrame([row])[FEATURES]


def score(y_true, y_pred):
    return {
        'MAE': round(float(mean_absolute_error(y_true, y_pred)), 2),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
    }
