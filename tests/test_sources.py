"""Live data sources and the real-lines tracker, with every network call replaced by fixtures."""

from datetime import date

import pandas as pd
import pytest

from nba_predictor import cdn, lines
from nba_predictor.injuries import match
from nba_predictor.store import fold


@pytest.mark.parametrize(("raw", "minutes"), [("PT25M46.00S", 26), ("PT05M10.00S", 5), ("PT00M00.00S", 0), ("", 0)])
def test_parse_minutes(raw, minutes):
    assert cdn.parse_minutes(raw) == minutes


def _player(pid, name, played="1", pts=10):
    stats = {
        k: 0
        for k in (
            "fieldGoalsMade",
            "fieldGoalsAttempted",
            "threePointersMade",
            "threePointersAttempted",
            "freeThrowsMade",
            "freeThrowsAttempted",
            "reboundsOffensive",
            "reboundsDefensive",
            "reboundsTotal",
            "assists",
            "steals",
            "blocks",
            "turnovers",
            "foulsPersonal",
            "plusMinusPoints",
        )
    }
    return {
        "personId": pid,
        "name": name,
        "played": played,
        "statistics": {**stats, "points": pts, "minutes": "PT30M00.00S"},
    }


def test_boxscore_rows_match_league_game_log(monkeypatch):
    game = {
        "game": {
            "homeTeam": {"teamId": 1, "teamTricode": "DEN", "score": 110, "players": [_player(7, "A", pts=30)]},
            "awayTeam": {
                "teamId": 2,
                "teamTricode": "BOS",
                "score": 100,
                "players": [_player(8, "B"), _player(9, "Bench", played="0")],
            },
        }
    }
    monkeypatch.setattr(cdn, "_get", lambda url, timeout=30: game)
    rows = cdn.boxscore_rows("0022600001", date(2026, 11, 18), "2026-27")
    assert [r["PLAYER_ID"] for r in rows] == [7, 8]  # the player who did not play is skipped
    assert rows[0]["MATCHUP"] == "DEN vs. BOS" and rows[0]["WL"] == "W" and rows[0]["MIN"] == 30
    assert rows[1]["MATCHUP"] == "BOS @ DEN" and rows[1]["SEASON_ID"] == "22026"


def test_update_season_refetches_unfinished_games(monkeypatch):
    fetched = []
    monkeypatch.setattr(
        cdn, "boxscore_rows", lambda gid, d, s: fetched.append(gid) or [{"GAME_ID": gid, "PLAYER_ID": 1}]
    )
    cached = pd.DataFrame({"GAME_ID": ["22600001", "22600002"], "PLAYER_ID": [1, 1]})
    schedule = [
        {"game_id": "0022600001", "game_date": date(2026, 11, 1), "status": "final"},
        {"game_id": "0022600002", "game_date": date(2026, 11, 2), "status": "final"},
        {"game_id": "0022600003", "game_date": date(2026, 11, 3), "status": "final"},
        {"game_id": "0022600004", "game_date": date(2026, 11, 4), "status": "live"},
    ]
    out = cdn.update_season(cached, "2026-27", schedule, pause=0, recheck_days=0)
    assert fetched == ["0022600003"]
    assert len(out) == 3


def test_injury_names_match_across_accents_and_teams():
    players = pd.DataFrame(
        {"name": ["Nikola Jokić", "Jalen Williams", "Jalen Williams"], "team": ["DEN", "OKC", "CHI"]},
        index=[203999, 1, 2],
    )
    report = [
        {"name": "Nikola Jokic", "team": "DEN", "status": "Out", "note": "", "return_date": None},
        {"name": "Jalen Williams", "team": "CHI", "status": "Day-To-Day", "note": "", "return_date": None},
        {"name": "Unknown Guy", "team": "LAL", "status": "Out", "note": "", "return_date": None},
    ]
    injuries = match(report, players, fold)
    assert set(injuries) == {203999, 2}
    assert injuries[203999].out and not injuries[2].out


EVENT = {
    "id": "evt1",
    "commence_time": "2026-11-19T00:00:00Z",
    "bookmakers": [
        {
            "key": "dk",
            "markets": [
                {
                    "key": "player_points",
                    "outcomes": [
                        {"name": "Over", "description": "Nikola Jokic", "price": -115, "point": 25.5},
                        {"name": "Under", "description": "Nikola Jokic", "price": -105, "point": 25.5},
                    ],
                }
            ],
        },
        {
            "key": "fd",
            "markets": [
                {
                    "key": "player_points",
                    "outcomes": [
                        {"name": "Over", "description": "Nikola Jokic", "price": -110, "point": 25.5},
                        {"name": "Under", "description": "Nikola Jokic", "price": -110, "point": 25.5},
                    ],
                }
            ],
        },
        {
            "key": "mgm",
            "markets": [
                {
                    "key": "player_points",
                    "outcomes": [
                        {"name": "Over", "description": "Nikola Jokic", "price": 100, "point": 26.5},
                        {"name": "Under", "description": "Nikola Jokic", "price": -120, "point": 26.5},
                    ],
                }
            ],
        },
    ],
}


def test_props_use_the_consensus_line_and_best_prices():
    [row] = lines.parse_event(EVENT)
    assert row["stat"] == "PTS" and row["line"] == 25.5
    assert row["over_odds"] == -110 and row["under_odds"] == -105  # best price at the consensus line
    assert row["books"] == 2


def test_grading_and_record():
    ledger = pd.DataFrame(
        [
            {
                "game_date": "2026-11-18",
                "player_id": 1,
                "stat": "PTS",
                "line": 20.5,
                "pick": "over",
                "pick_odds": -110,
                "p_over": 0.6,
                "result": None,
                "actual": None,
                "profit": None,
            },
            {
                "game_date": "2026-11-18",
                "player_id": 2,
                "stat": "PTS",
                "line": 10.5,
                "pick": "under",
                "pick_odds": 100,
                "p_over": 0.4,
                "result": None,
                "actual": None,
                "profit": None,
            },
            {
                "game_date": "2026-11-18",
                "player_id": 3,
                "stat": "PTS",
                "line": 15.5,
                "pick": "pass",
                "pick_odds": None,
                "p_over": 0.5,
                "result": None,
                "actual": None,
                "profit": None,
            },
            {
                "game_date": "2026-11-20",
                "player_id": 1,
                "stat": "PTS",
                "line": 20.5,
                "pick": "over",
                "pick_odds": -110,
                "p_over": 0.6,
                "result": None,
                "actual": None,
                "profit": None,
            },
        ]
    )
    logs = pd.DataFrame({"PLAYER_ID": [1, 2, 3], "GAME_DATE": ["2026-11-18"] * 3, "PTS": [25, 14, 9]})
    graded = lines.grade(ledger, logs)
    assert graded["result"].tolist()[:3] == ["over", "over", "under"]
    assert graded["profit"].iloc[0] == pytest.approx(100 / 110)
    assert graded["profit"].iloc[1] == -1.0
    assert pd.isna(graded["profit"].iloc[2]) and pd.isna(graded["result"].iloc[3])
    s = lines.summary(graded)
    assert (s["picks"], s["wins"], s["losses"], s["pending"]) == (2, 1, 1, 1)


def test_recent_final_games_are_refetched(monkeypatch):
    """A game cached while live is replaced once final, as long as it is recent."""
    fetched = []
    monkeypatch.setattr(cdn, "boxscore_rows", lambda gid, d, s: fetched.append(gid) or [{"GAME_ID": gid}])
    cached = pd.DataFrame({"GAME_ID": ["22600001", "22600002"], "PLAYER_ID": [1, 1]})
    schedule = [
        {"game_id": "0022600001", "game_date": date(2026, 11, 1), "status": "final"},
        {"game_id": "0022600002", "game_date": date(2026, 11, 18), "status": "final"},
    ]
    cdn.update_season(cached, "2026-27", schedule, pause=0, recheck_days=3)
    assert fetched == ["0022600002"]
