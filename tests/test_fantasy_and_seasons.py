from datetime import date

import numpy as np
import pytest

from nba_predictor.fantasy import PRESETS, doubles, score
from nba_predictor.seasons import current_season, previous_season, season_id

LINE = {
    "PTS": 30,
    "REB": 10,
    "AST": 8,
    "FG3M": 4,
    "STL": 2,
    "BLK": 1,
    "TOV": 3,
    "FGM": 11,
    "FGA": 20,
    "FTM": 4,
    "FTA": 5,
}


def scored(key: str, line: dict = LINE) -> float:
    return float(score({k: np.array([v]) for k, v in line.items()}, PRESETS[key])[0])


def test_espn_points_league():
    # 30 + 2*11 - 20 + 4 - 5 + 4 + 10 + 2*8 + 4*2 + 4*1 - 2*3
    assert scored("espn") == pytest.approx(67)


def test_yahoo_points_league():
    assert scored("yahoo") == pytest.approx(30 + 12 + 12 + 6 + 3 - 3)


def test_sleeper_bonuses():
    base = 0.5 * 30 + 0.5 * 4 + 10 + 8 + 2 * 2 + 2 * 1 - 3
    assert scored("sleeper") == pytest.approx(base + 1)  # double-double bonus
    triple = {**LINE, "AST": 10, "PTS": 41}
    expected = 0.5 * 41 + 0.5 * 4 + 10 + 10 + 4 + 2 - 3 + 1 + 2 + 2  # DD + TD + 40 point bonus
    assert scored("sleeper", triple) == pytest.approx(expected)


def test_doubles_counts_double_figure_categories():
    stats = {
        "PTS": np.array([10, 9]),
        "REB": np.array([10, 12]),
        "AST": np.array([10, 1]),
        "STL": np.array([0, 0]),
        "BLK": np.array([0, 10]),
    }
    assert doubles(stats).tolist() == [3, 2]


@pytest.mark.parametrize(
    ("today", "season"),
    [
        (date(2026, 11, 18), "2026-27"),
        (date(2026, 9, 30), "2025-26"),
        (date(2026, 10, 1), "2026-27"),
        (date(2027, 4, 10), "2026-27"),
    ],
)
def test_current_season_from_date(today, season):
    assert current_season(today) == season


def test_previous_season_and_id():
    assert previous_season("2026-27") == "2025-26"
    assert previous_season("2000-01") == "1999-00"
    assert season_id("2025-26") == "22025"
