"""NBA season arithmetic. Seasons start in October, so November 2026 is the 2026-27 season."""

from __future__ import annotations

from datetime import date


def season_start_year(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 10 else today.year - 1


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def current_season(today: date | None = None) -> str:
    return season_label(season_start_year(today))


def previous_season(season: str) -> str:
    return season_label(int(season[:4]) - 1)


def season_id(season: str) -> str:
    """stats.nba.com regular season SEASON_ID, e.g. '22025' for 2025-26."""
    return "2" + season[:4]
