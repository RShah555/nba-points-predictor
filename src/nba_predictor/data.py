"""Downloading and caching league wide player game logs from stats.nba.com."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
from tenacity import Retrying, before_sleep_log, stop_after_attempt, wait_fixed

from .config import Settings, get_settings

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = frozenset(
    {
        "SEASON_ID",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "GAME_ID",
        "GAME_DATE",
        "MATCHUP",
        "MIN",
        "FGM",
        "FGA",
        "FG3M",
        "FG3A",
        "FTM",
        "FTA",
        "OREB",
        "DREB",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "PTS",
        "PLUS_MINUS",
    }
)


class SeasonUnavailableError(RuntimeError):
    """stats.nba.com returned no games for a season."""


class SchemaError(ValueError):
    """A game log is missing columns the pipeline needs."""


def validate(df: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SchemaError(f"{source} is missing columns: {sorted(missing)}")
    return df


def season_path(season: str, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.data_dir / f"player_logs_{season}.csv"


def download_season(season: str, settings: Settings | None = None) -> pd.DataFrame:
    """One LeagueGameLog call returns every player game of the regular season."""
    from nba_api.stats.endpoints import leaguegamelog

    settings = settings or get_settings()
    retrying = Retrying(
        stop=stop_after_attempt(settings.retry_attempts),
        wait=wait_fixed(settings.retry_wait_seconds),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    for attempt in retrying:
        with attempt:
            frames = leaguegamelog.LeagueGameLog(
                season=season,
                player_or_team_abbreviation="P",
                season_type_all_star="Regular Season",
                timeout=settings.request_timeout,
            ).get_data_frames()
    df = frames[0]
    if df.empty:
        raise SeasonUnavailableError(f"stats.nba.com has no regular season games for {season}")
    return validate(df, f"LeagueGameLog {season}")


def _is_stale(path: Path, max_age_hours: float | None) -> bool:
    return max_age_hours is not None and (time.time() - path.stat().st_mtime) > max_age_hours * 3600


def load_season(season: str, *, max_age_hours: float | None = None, settings: Settings | None = None) -> pd.DataFrame:
    """Read a season from the local cache, downloading it when missing or stale.

    Completed seasons never change, so they are cached forever (max_age_hours=None).
    The season in progress is refreshed when the cached copy is older than max_age_hours;
    if that refresh fails the stale copy is used rather than failing outright.
    """
    settings = settings or get_settings()
    path = season_path(season, settings)
    if path.exists() and not _is_stale(path, max_age_hours):
        return validate(pd.read_csv(path), str(path))
    try:
        log.info("Downloading %s player game logs", season)
        df = download_season(season, settings)
    except SeasonUnavailableError:
        raise
    except Exception:
        if path.exists():
            log.exception("Refreshing %s failed, using the cached copy", season)
            return validate(pd.read_csv(path), str(path))
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    time.sleep(1)  # be polite to stats.nba.com
    return df
