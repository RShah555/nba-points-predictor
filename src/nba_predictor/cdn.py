"""NBA's public CDN: the season schedule and per game box scores.

stats.nba.com blocks most cloud servers, but cdn.nba.com does not, so the hosted app keeps
the season in progress up to date from here: every final game's box score is turned into
LeagueGameLog shaped rows and appended to the local cache.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date
from typing import Any

import httpx
import pandas as pd
from nba_api.live.nba.library.http import NBALiveHTTP

from .seasons import season_id

log = logging.getLogger(__name__)

BASE = "https://cdn.nba.com/static/json"
# The CDN rejects requests that do not look like a browser on nba.com; nba_api maintains a header set it accepts.
HEADERS = {k: v for k, v in NBALiveHTTP.headers.items() if k.lower() != "host"}
STATUS = {1: "scheduled", 2: "live", 3: "final"}


def _get(url: str, timeout: float = 30) -> dict[str, Any]:
    r = httpx.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def schedule_rows() -> list[dict[str, Any]]:
    """Every regular season game of the current season: id, date, teams, status, tip off."""
    data = _get(f"{BASE}/staticData/scheduleLeagueV2.json", timeout=60)
    rows = []
    for day in data["leagueSchedule"]["gameDates"]:
        for g in day["games"]:
            if not str(g["gameId"]).startswith("002"):  # regular season only
                continue
            rows.append(
                {
                    "game_id": str(g["gameId"]),
                    "game_date": pd.Timestamp(g["gameDateEst"]).date(),
                    "home": g["homeTeam"]["teamTricode"],
                    "away": g["awayTeam"]["teamTricode"],
                    "status": STATUS.get(int(g["gameStatus"]), "scheduled"),
                    "tipoff_utc": g.get("gameDateTimeUTC"),
                }
            )
    return rows


_MINUTES = re.compile(r"PT(?:(\d+)M)?(?:([\d.]+)S)?")


def parse_minutes(value: str) -> int:
    """'PT25M46.00S' -> 26 (LeagueGameLog reports whole minutes)."""
    m = _MINUTES.fullmatch(value or "")
    if not m:
        return 0
    return round(int(m.group(1) or 0) + float(m.group(2) or 0) / 60)


def boxscore_rows(game_id: str, game_date: date, season: str) -> list[dict[str, Any]]:
    """LeagueGameLog shaped rows for every player who played in a final game."""
    game = _get(f"{BASE}/liveData/boxscore/boxscore_{game_id}.json")["game"]
    home, away = game["homeTeam"], game["awayTeam"]
    rows = []
    for team, opp, is_home in ((home, away, True), (away, home, False)):
        won = team["score"] > opp["score"]
        matchup = (
            f"{team['teamTricode']} vs. {opp['teamTricode']}"
            if is_home
            else (f"{team['teamTricode']} @ {opp['teamTricode']}")
        )
        for p in team["players"]:
            if str(p.get("played")) != "1":
                continue
            s = p["statistics"]
            rows.append(
                {
                    "SEASON_ID": season_id(season),
                    "PLAYER_ID": p["personId"],
                    "PLAYER_NAME": p["name"],
                    "TEAM_ID": team["teamId"],
                    "TEAM_ABBREVIATION": team["teamTricode"],
                    "GAME_ID": game_id,
                    "GAME_DATE": game_date.isoformat(),
                    "MATCHUP": matchup,
                    "WL": "W" if won else "L",
                    "MIN": parse_minutes(s["minutes"]),
                    "FGM": s["fieldGoalsMade"],
                    "FGA": s["fieldGoalsAttempted"],
                    "FG3M": s["threePointersMade"],
                    "FG3A": s["threePointersAttempted"],
                    "FTM": s["freeThrowsMade"],
                    "FTA": s["freeThrowsAttempted"],
                    "OREB": s["reboundsOffensive"],
                    "DREB": s["reboundsDefensive"],
                    "REB": s["reboundsTotal"],
                    "AST": s["assists"],
                    "STL": s["steals"],
                    "BLK": s["blocks"],
                    "TOV": s["turnovers"],
                    "PF": s["foulsPersonal"],
                    "PTS": s["points"],
                    "PLUS_MINUS": s["plusMinusPoints"],
                }
            )
    return rows


def update_season(
    cached: pd.DataFrame | None,
    season: str,
    schedule: list[dict[str, Any]],
    pause: float = 0.2,
    recheck_days: int = 3,
) -> pd.DataFrame:
    """Append box scores for final games the cache does not have yet.

    Cached rows for games that are not final, or that finished in the last `recheck_days`, are
    dropped and fetched fresh: a download taken mid game (stats.nba.com lists live games) would
    otherwise leave a half played box score in the cache after the game ends.
    """
    if cached is not None and not cached.empty:
        latest = max((g["game_date"] for g in schedule if g["status"] == "final"), default=None)
        stale = {
            g["game_id"].lstrip("0")
            for g in schedule
            if g["status"] != "final" or (latest is not None and (latest - g["game_date"]).days < recheck_days)
        }
        cached = cached[~cached["GAME_ID"].astype(str).str.lstrip("0").isin(stale)]
    have = set() if cached is None or cached.empty else set(cached["GAME_ID"].astype(str).str.lstrip("0"))
    missing = [g for g in schedule if g["status"] == "final" and g["game_id"].lstrip("0") not in have]
    if missing:
        log.info("Fetching %d new box scores from cdn.nba.com", len(missing))
    rows: list[dict[str, Any]] = []
    for g in missing:
        try:
            rows += boxscore_rows(g["game_id"], g["game_date"], season)
        except Exception:
            log.exception("Box score %s failed; it will be retried next refresh", g["game_id"])
        time.sleep(pause)
    new = pd.DataFrame(rows)
    if cached is None or cached.empty:
        return new
    return pd.concat([cached, new], ignore_index=True) if len(new) else cached
