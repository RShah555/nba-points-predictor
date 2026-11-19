"""NBA Points Predictor: projects a player's points in his next game.

    streamlit run app.py
"""
import os
from datetime import date

import joblib
import pandas as pd
import streamlit as st
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players, teams

from model import STATS, next_game_features, normalize

MODEL_PATH = os.path.join('models', 'points_model.joblib')
CACHE_SECONDS = 3600


def current_season(today=None):
    """NBA seasons start in October, so Nov 2026 is the 2026-27 season."""
    today = today or date.today()
    start = today.year if today.month >= 10 else today.year - 1
    return f'{start}-{str(start + 1)[-2:]}'


def previous_season(season):
    start = int(season[:4]) - 1
    return f'{start}-{str(start + 1)[-2:]}'


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@st.cache_data(ttl=CACHE_SECONDS)
def active_players():
    return {p['full_name']: p['id'] for p in sorted(players.get_active_players(), key=lambda p: p['full_name'])}


@st.cache_data(ttl=CACHE_SECONDS)
def game_log(player_id, season):
    try:
        return playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=60).get_data_frames()[0]
    except Exception as e:  # stats.nba.com is slow or rate limited sometimes
        st.warning(f'Could not load {season} games: {e}')
        return pd.DataFrame()


def recent_games(player_id):
    season = current_season()
    logs = [game_log(player_id, s) for s in (previous_season(season), season)]
    logs = [df for df in logs if not df.empty]
    return pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()


def projection_section(log, bundle, opponent, home, rest_days):
    features = next_game_features(log, home, rest_days, bundle['defense'].get(opponent))
    projected = float(bundle['model'].predict(features)[0])
    results = bundle['metrics']['results']

    c1, c2, c3 = st.columns(3)
    c1.metric('Projected points', f'{projected:.1f}')
    c2.metric('Last 5 games', f"{features['PTS_roll_5'].iloc[0]:.1f}")
    c3.metric('Season average', f"{features['SEASON_AVG_PTS'].iloc[0]:.1f}")
    st.caption(
        f"{'Home' if home else 'Away'} vs {opponent} on {rest_days} day(s) of rest. "
        f"On the held out {bundle['metrics']['test_season']} season the model missed by "
        f"{results['Gradient boosting model']['MAE']} points per game on average, versus "
        f"{results['Season average so far']['MAE']} for the season average and "
        f"{results['Last 5 game average']['MAE']} for the last 5 game average."
    )


def form_section(log):
    df = normalize(log)
    df['5 game average'] = df['PTS'].rolling(5, min_periods=1).mean()
    st.subheader('Scoring trend')
    st.line_chart(df.set_index('GAME_DATE')[['PTS', '5 game average']].rename(columns={'PTS': 'Points'}))

    st.subheader('Last 5 games')
    last5 = df.tail(5).iloc[::-1]
    last5 = last5.assign(GAME_DATE=last5['GAME_DATE'].dt.strftime('%b %d, %Y'))
    st.dataframe(last5[['GAME_DATE', 'MATCHUP', 'MIN', 'PTS', 'REB', 'AST', 'FG_PCT']], hide_index=True)

    st.subheader('Averages over the last two seasons')
    cols = st.columns(len(STATS))
    for col, stat in zip(cols, STATS):
        col.metric(stat, f'{df[stat].mean():.1f}')


def main():
    st.set_page_config(page_title='NBA Points Predictor', page_icon='🏀', layout='wide')
    st.title('NBA Points Predictor 🏀')
    st.write("Projects a player's points in his next game from his recent form, the opponent's defense, home court and rest.")

    names = active_players()
    team_abbrevs = sorted(t['abbreviation'] for t in teams.get_teams())

    with st.sidebar:
        player = st.selectbox('Player', list(names), index=None, placeholder='Search for a player')
        opponent = st.selectbox('Next opponent', team_abbrevs)
        home = st.radio('Location', ['Home', 'Away'], horizontal=True) == 'Home'
        rest_days = st.slider('Days of rest', 1, 7, 2)

    if not player:
        st.info('Pick a player in the sidebar to get started.')
        return

    with st.spinner(f'Loading games for {player}...'):
        log = recent_games(names[player])
    if log.empty:
        st.error('No games found for this player in the last two seasons.')
        return

    bundle = load_model()
    st.subheader(f'{player}: next game projection')
    if bundle is None:
        st.info('No trained model found. Run `python train_model.py` once, then reload this page.')
    else:
        projection_section(log, bundle, opponent, home, rest_days)

    form_section(log)


if __name__ == '__main__':
    main()
