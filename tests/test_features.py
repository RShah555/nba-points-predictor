"""No feature may ever use the game it is predicting, and live features must equal training features."""

import numpy as np
import pandas as pd
import pytest

from conftest import fake_logs
from nba_predictor.features import (
    ABSENCE_FEATURES,
    DEFENSE_FEATURES,
    FEATURES,
    GameContext,
    add_player_features,
    build_training_frame,
    defense_features,
    next_game_features,
    normalize,
    roster_snapshot,
)


def test_rolling_features_only_use_past_games():
    df = add_player_features(normalize(fake_logs([10, 20, 30, 40, 50, 60])))
    # Before game 4 (40 points), the last three games scored 10, 20 and 30
    assert df.loc[3, "PTS_roll_3"] == 20
    assert df.loc[3, "SEASON_AVG_PTS"] == 20
    # The first game has no history at all
    assert pd.isna(df.loc[0, "PTS_roll_3"])
    assert pd.isna(df.loc[0, "PTS_ewm"])


def test_changing_a_game_never_changes_its_own_features():
    a = add_player_features(normalize(fake_logs([10, 20, 30, 40, 50, 60])))
    b = add_player_features(normalize(fake_logs([10, 20, 30, 99, 50, 60])))
    player_features = [c for c in FEATURES if c not in DEFENSE_FEATURES and c not in ABSENCE_FEATURES]
    pd.testing.assert_frame_equal(a.loc[:3, player_features], b.loc[:3, player_features])
    assert not a.loc[4, player_features].equals(b.loc[4, player_features])


def test_home_rest_and_schedule():
    df = add_player_features(normalize(fake_logs([10] * 6)))
    assert df["HOME"].tolist() == [1, 0, 1, 0, 1, 0]
    assert df.loc[1, "REST_DAYS"] == 2
    assert df.loc[0, "REST_DAYS"] == 3  # no previous game: assume normal rest
    # games every 2 days: the previous 7 days hold the last 3 games (days -2, -4, -6)
    assert df.loc[5, "GAMES_LAST_7D"] == 3


def test_previous_season_features_come_from_the_whole_prior_season():
    raw = pd.concat([fake_logs([10] * 6, "22024", "2024-10-22"), fake_logs([30] * 6, "22025", "2025-10-22")])
    df = add_player_features(normalize(raw))
    current = df[df["SEASON_ID"] == "22025"]
    assert (current["PREV_SEASON_PTS"] == 10).all()
    assert (current["PREV_SEASON_GP"] == 6).all()
    assert df[df["SEASON_ID"] == "22024"]["PREV_SEASON_PTS"].isna().all()


def test_defense_features_are_shifted(league):
    norm = normalize(league)
    d = defense_features(norm)
    first = d.groupby("DEF_TEAM").head(1)
    assert first["OPP_PTS_ALLOWED"].isna().all()  # nothing allowed yet before a team's first game


def test_training_frame_has_every_feature(league):
    df, snapshot = build_training_frame(league)
    assert set(FEATURES) <= set(df.columns)
    assert (df["GAMES_PLAYED"] >= 5).all()
    assert list(snapshot.columns) == DEFENSE_FEATURES


def test_live_features_match_training_features(league):
    """Train/serve parity: predicting a played game from the games before it gives the training row."""
    frame, _ = build_training_frame(league)
    target = frame[frame["SEASON_ID"] == "22025"].iloc[200]
    pid, game_date = int(target["PLAYER_ID"]), target["GAME_DATE"]
    before = league[pd.to_datetime(league["GAME_DATE"]) < game_date]
    ctx = GameContext(pid, target["OPP"], bool(target["HOME"]), game_date.date(), "22025")
    player_features = [c for c in FEATURES if c not in DEFENSE_FEATURES and c not in ABSENCE_FEATURES]
    live = next_game_features(before, [ctx], pd.DataFrame(columns=DEFENSE_FEATURES))
    np.testing.assert_allclose(
        live[player_features].iloc[0].to_numpy(dtype=float),
        target[player_features].to_numpy(dtype=float),
        equal_nan=True,
    )


def _without(league, player_id, game_id):
    return league[~((league["PLAYER_ID"] == player_id) & (league["GAME_ID"] == game_id))]


def test_absent_regular_is_counted_for_teammates(league):
    """Drop a regular from one game: his teammates' OUT_* features pick him up, nobody else's do."""
    games = league[(league["PLAYER_ID"] == 0) & (league["SEASON_ID"] == "22025")]
    game_id = games["GAME_ID"].iloc[15]
    frame, _ = build_training_frame(_without(league, 0, game_id))
    rows = frame[frame["GAME_ID"] == game_id]
    teammates = rows[rows["TEAM"] == "BOS"]
    opponents = rows[rows["TEAM"] != "BOS"]
    baseline, _ = build_training_frame(league)
    before = baseline[(baseline["GAME_ID"] == game_id) & (baseline["TEAM"] == "BOS")]
    assert (teammates["OUT_COUNT"].to_numpy() == before["OUT_COUNT"].iloc[0] + 1).all()
    assert (teammates["OUT_MIN"] > before["OUT_MIN"].iloc[0]).all()
    assert opponents["OUT_COUNT"].equals(
        baseline[(baseline["GAME_ID"] == game_id) & (baseline["TEAM"] != "BOS")].set_index(opponents.index)["OUT_COUNT"]
    )


def test_live_absences_match_training(league):
    """Train/serve parity for teammates out: the injury report path gives the box score path's numbers."""
    games = league[(league["PLAYER_ID"] == 0) & (league["SEASON_ID"] == "22025")]
    game_id, game_date = games["GAME_ID"].iloc[15], pd.Timestamp(games["GAME_DATE"].iloc[15])
    data = _without(league, 0, game_id)
    frame, _ = build_training_frame(data)
    target = frame[(frame["GAME_ID"] == game_id) & (frame["PLAYER_ID"] == 1)].iloc[0]
    before = data[pd.to_datetime(data["GAME_DATE"]) < game_date]
    ctx = GameContext(1, target["OPP"], bool(target["HOME"]), game_date.date(), "22025", "BOS", teammates_out=(0,))
    live = next_game_features(before, [ctx], pd.DataFrame(columns=DEFENSE_FEATURES), roster_snapshot(normalize(before)))
    np.testing.assert_allclose(
        live[ABSENCE_FEATURES].iloc[0].to_numpy(dtype=float), target[ABSENCE_FEATURES].to_numpy(dtype=float)
    )


def test_next_game_handles_the_same_player_twice(league):
    pid = int(league["PLAYER_ID"].iloc[0])
    last = pd.to_datetime(league["GAME_DATE"]).max().date()
    ctxs = [GameContext(pid, "NYK", True, last + pd.Timedelta(days=d), "22025") for d in (1, 3)]
    rows = next_game_features(league, ctxs, pd.DataFrame(columns=DEFENSE_FEATURES))
    assert rows["REST_DAYS"].tolist() == [1, 3]
    assert rows["BACK_TO_BACK"].tolist() == [1, 0]


def test_unknown_player_raises(league):
    with pytest.raises(KeyError):
        next_game_features(
            league,
            [GameContext(999_999, "NYK", True, pd.Timestamp("2026-01-01").date(), "22025")],
            pd.DataFrame(columns=DEFENSE_FEATURES),
        )
