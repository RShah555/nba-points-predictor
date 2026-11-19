"""Live league data for serving: last season plus the season in progress, and the schedule."""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

import pandas as pd

from .config import Settings, get_settings
from .data import SeasonUnavailableError, load_season
from .features import defense_snapshot, normalize
from .seasons import current_season, previous_season

log = logging.getLogger(__name__)


def fold(text: str) -> str:
    """Lowercase and strip accents so 'jokic' finds 'Nikola Jokić' and 'doncic' finds 'Luka Dončić'."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower().strip()


@dataclass(frozen=True)
class ScheduledGame:
    game_id: str
    game_date: date
    home: str
    away: str
    status: str
    tipoff_utc: str | None


class LeagueStore:
    """Thread safe holder for the data the API serves from. Refreshes itself in the background."""

    def __init__(
        self,
        settings: Settings | None = None,
        raw: pd.DataFrame | None = None,
        schedule: list[ScheduledGame] | None = None,
        today: date | None = None,
    ):
        self.settings = settings or get_settings()
        self._today = today
        self._lock = threading.Lock()
        self._refreshing = False
        self.version = 0
        self.loaded_at = 0.0
        self.seasons: list[str] = []
        self._schedule = schedule or []
        if raw is not None:
            self._install(raw)

    @property
    def today(self) -> date:
        return self._today or date.today()

    @property
    def season(self) -> str:
        return current_season(self.today)

    # loading -----------------------------------------------------------------
    def load(self) -> None:
        season = self.season
        frames, loaded = [], []
        for s in (previous_season(season), season):
            try:
                live = s == season
                frames.append(
                    load_season(
                        s, max_age_hours=self.settings.live_refresh_hours if live else None, settings=self.settings
                    )
                )
                loaded.append(s)
            except SeasonUnavailableError:
                log.warning("No games yet for %s", s)
        if not frames:
            raise SeasonUnavailableError("No league data available for the last two seasons")
        try:
            self._schedule = load_schedule(season, self.settings)
        except Exception:
            log.exception("Could not load the %s schedule; next games must be entered by hand", season)
        self._install(pd.concat(frames, ignore_index=True), loaded)

    def completed_games(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Drop rows for games that are not over yet.

        LeagueGameLog lists games that are live (with partial box scores) or not started (all zeros).
        Those rows would leak a half played game into the features, so only games the schedule marks
        final are kept; without a schedule, anything dated today or later is dropped.
        """
        game_ids = raw["GAME_ID"].astype(str).str.lstrip("0")
        unfinished = {g.game_id.lstrip("0") for g in self._schedule if g.status != "final"}
        keep = ~game_ids.isin(unfinished)
        if not self._schedule:
            keep &= pd.to_datetime(raw["GAME_DATE"], format="mixed").dt.date < self.today
        dropped = int((~keep).sum())
        if dropped:
            log.info("Ignoring %d player rows from games that are not final", dropped)
        return raw[keep]

    def _install(self, raw: pd.DataFrame, seasons: list[str] | None = None) -> None:
        raw = self.completed_games(raw)
        norm = normalize(raw)
        players = self._player_table(norm)
        snapshot = defense_snapshot(norm)
        with self._lock:
            self.raw, self.norm, self.players, self.snapshot = raw, norm, players, snapshot
            self.seasons = seasons or sorted(
                {f"{sid[1:]}-{str(int(sid[1:]) + 1)[-2:]}" for sid in norm["SEASON_ID"].unique()}
            )
            self.version += 1
            self.loaded_at = time.time()

    def refresh_if_stale(self) -> None:
        stale = time.time() - self.loaded_at > self.settings.live_refresh_hours * 3600
        if not stale or self._refreshing or not self.loaded_at:
            return
        self._refreshing = True

        def work() -> None:
            try:
                self.load()
            except Exception:
                log.exception("Background refresh failed")
            finally:
                self._refreshing = False

        threading.Thread(target=work, daemon=True).start()

    # queries -----------------------------------------------------------------
    @staticmethod
    def _player_table(norm: pd.DataFrame) -> pd.DataFrame:
        latest_season = norm["SEASON_ID"].max()
        last = norm.groupby("PLAYER_ID").tail(1).set_index("PLAYER_ID")
        recent = norm.groupby("PLAYER_ID").tail(10).groupby("PLAYER_ID")
        current = norm[norm["SEASON_ID"] == latest_season].groupby("PLAYER_ID")
        table = pd.DataFrame(
            {
                "name": last["PLAYER_NAME"],
                "team": last["TEAM"],
                "team_id": last.get("TEAM_ID"),
                "last_game": last["GAME_DATE"],
                "min_last_10": recent["MIN"].mean(),
                "games": current.size(),
                "pts": current["PTS"].mean(),
                "reb": current["REB"].mean(),
                "ast": current["AST"].mean(),
                "min": current["MIN"].mean(),
            }
        )
        table["games"] = table["games"].fillna(0).astype(int)
        return table.sort_values("name")

    def search(self, query: str = "", limit: int = 20) -> pd.DataFrame:
        table = self.players
        if query:
            q = fold(query)
            names = table["name"].map(fold)
            mask = names.str.contains(q, regex=False)
            hits, folded = table[mask], names[mask]
            starts = folded.str.startswith(q) | folded.str.contains(" " + q, regex=False)
            hits = hits.assign(_rank=(~starts).astype(int), _min=-hits["min_last_10"].fillna(0))
            return hits.sort_values(["_rank", "_min"]).drop(columns=["_rank", "_min"]).head(limit)
        return table.sort_values("min_last_10", ascending=False).head(limit)

    def player(self, player_id: int) -> pd.Series:
        if player_id not in self.players.index:
            raise KeyError(f"Unknown player {player_id}")
        return cast(pd.Series, self.players.loc[player_id])

    def history(self, player_ids: list[int] | int) -> pd.DataFrame:
        ids = [player_ids] if isinstance(player_ids, int) else player_ids
        return self.raw[self.raw["PLAYER_ID"].isin(ids)]

    def game_log(self, player_id: int) -> pd.DataFrame:
        return self.norm[self.norm["PLAYER_ID"] == player_id]

    def next_game(self, team: str, after: date | None = None) -> ScheduledGame | None:
        after = after or self.today
        upcoming = [
            g for g in self._schedule if team in (g.home, g.away) and g.game_date >= after and g.status != "final"
        ]
        return min(upcoming, key=lambda g: g.game_date) if upcoming else None

    def slate(self, on: date | None = None) -> tuple[date | None, list[ScheduledGame]]:
        """All games on the first date (from `on`) that still has a game not yet final."""
        on = on or self.today
        future = sorted({g.game_date for g in self._schedule if g.game_date >= on and g.status != "final"})
        if not future:
            return None, []
        day = future[0]
        return day, [g for g in self._schedule if g.game_date == day]


def load_schedule(season: str, settings: Settings) -> list[ScheduledGame]:
    path = settings.data_dir / f"schedule_{season}.csv"
    fresh = path.exists() and time.time() - path.stat().st_mtime < settings.live_refresh_hours * 3600
    if fresh:
        df = pd.read_csv(path, dtype={"gameId": str})
    else:
        from nba_api.stats.endpoints import scheduleleaguev2

        df = scheduleleaguev2.ScheduleLeagueV2(season=season, timeout=settings.request_timeout).get_data_frames()[0]
        cols = [
            "gameId",
            "gameDateEst",
            "gameDateTimeUTC",
            "gameStatus",
            "homeTeam_teamTricode",
            "awayTeam_teamTricode",
        ]
        df = df[cols]
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    df = df[df["gameId"].astype(str).str.lstrip("0").str.startswith("2")]  # regular season only
    status = {1: "scheduled", 2: "live", 3: "final"}
    return [
        ScheduledGame(
            str(r["gameId"]),
            pd.Timestamp(r["gameDateEst"]).date(),
            r["homeTeam_teamTricode"],
            r["awayTeam_teamTricode"],
            status.get(int(r["gameStatus"]), "scheduled"),
            None if pd.isna(r["gameDateTimeUTC"]) else str(r["gameDateTimeUTC"]),
        )
        for r in cast(list[dict[str, Any]], df.to_dict("records"))
        if isinstance(r["homeTeam_teamTricode"], str) and isinstance(r["awayTeam_teamTricode"], str)
    ]
