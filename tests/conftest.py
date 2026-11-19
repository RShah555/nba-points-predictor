"""Shared fixtures: synthetic league data so tests never touch stats.nba.com."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

TEAMS = ["BOS", "NYK", "LAL", "GSW"]


def fake_logs(
    points: list[float], season_id: str = "22025", start: str = "2025-10-22", player_id: int = 1
) -> pd.DataFrame:
    """A single player's season, alternating home and away against NYK every other day."""
    n = len(points)
    return pd.DataFrame(
        {
            "SEASON_ID": [season_id] * n,
            "PLAYER_ID": [player_id] * n,
            "PLAYER_NAME": ["Test Player"] * n,
            "TEAM_ID": [1] * n,
            "TEAM_ABBREVIATION": ["BOS"] * n,
            "GAME_ID": [f"{season_id}{i:04d}" for i in range(n)],
            "GAME_DATE": pd.date_range(start, periods=n, freq="2D").strftime("%Y-%m-%d"),
            "MATCHUP": ["BOS vs. NYK" if i % 2 == 0 else "BOS @ NYK" for i in range(n)],
            "WL": ["W"] * n,
            "PTS": points,
            "MIN": [30] * n,
            "FGM": [6] * n,
            "FGA": [15] * n,
            "FG3M": [2] * n,
            "FG3A": [5] * n,
            "FTM": [3] * n,
            "FTA": [4] * n,
            "OREB": [1] * n,
            "DREB": [5] * n,
            "REB": [6] * n,
            "AST": [5] * n,
            "STL": [1] * n,
            "BLK": [1] * n,
            "TOV": [2] * n,
            "PF": [2] * n,
            "PLUS_MINUS": [0] * n,
        }
    )


def synthetic_league(
    seasons: tuple[str, ...] = ("22024", "22025"), games: int = 30, players_per_team: int = 6, seed: int = 0
) -> pd.DataFrame:
    """A small league: 4 teams in a round robin, every player with a stable scoring level plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    level = {(t, j): rng.uniform(4, 28) for t in TEAMS for j in range(players_per_team)}
    for season in seasons:
        start = date(int(season[1:]), 10, 22)
        for g in range(games):
            day = start + timedelta(days=2 * g)
            pairs = [(TEAMS[0], TEAMS[1 + g % 3]), tuple(t for t in TEAMS[1:] if t != TEAMS[1 + g % 3])]
            for home, away in pairs:
                game_id = f"{season}{g:03d}{home}"
                for team, opp, is_home in ((home, away, True), (away, home, False)):
                    for j in range(players_per_team):
                        mu = level[(team, j)]
                        minutes = float(np.clip(rng.normal(10 + mu, 4), 4, 44))
                        fga = rng.poisson(mu / 1.9)
                        fgm = rng.binomial(fga, 0.47)
                        fg3a = rng.poisson(fga * 0.35)
                        fg3m = rng.binomial(fg3a, 0.36)
                        fta = rng.poisson(mu / 6)
                        ftm = rng.binomial(fta, 0.78)
                        oreb, dreb = rng.poisson(1), rng.poisson(mu / 6)
                        rows.append(
                            {
                                "SEASON_ID": season,
                                "PLAYER_ID": TEAMS.index(team) * 100 + j,
                                "PLAYER_NAME": f"{team} Player {j}",
                                "TEAM_ID": TEAMS.index(team),
                                "TEAM_ABBREVIATION": team,
                                "GAME_ID": game_id,
                                "GAME_DATE": day.isoformat(),
                                "MATCHUP": f"{team} vs. {opp}" if is_home else f"{team} @ {opp}",
                                "WL": "W",
                                "MIN": round(minutes),
                                "FGM": fgm,
                                "FGA": fga,
                                "FG3M": fg3m,
                                "FG3A": fg3a,
                                "FTM": ftm,
                                "FTA": fta,
                                "OREB": oreb,
                                "DREB": dreb,
                                "REB": oreb + dreb,
                                "AST": rng.poisson(mu / 5),
                                "STL": rng.poisson(0.9),
                                "BLK": rng.poisson(0.5),
                                "TOV": rng.poisson(1.5),
                                "PF": rng.poisson(2),
                                "PTS": 2 * (fgm - fg3m) + 3 * fg3m + ftm,
                                "PLUS_MINUS": int(rng.integers(-15, 16)),
                            }
                        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def league() -> pd.DataFrame:
    return synthetic_league()
