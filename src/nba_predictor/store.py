"""Live league data for serving: last season, the season in progress, the schedule and injuries."""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import cast

import pandas as pd

from . import cdn
from .config import Settings, get_settings
from .data import SeasonUnavailableError, load_season, season_path, validate
from .features import REGULAR_DAYS, REGULAR_MINUTES, defense_snapshot, normalize, roster_snapshot
from .injuries import Injury, fetch_report, match
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
        injuries: dict[int, Injury] | None = None,
    ):
        self.settings = settings or get_settings()
        self._today = today
        self._lock = threading.Lock()
        self._refreshing = False
        self.version = 0
        self.loaded_at = 0.0
        self.injuries_at = 0.0
        self.seasons: list[str] = []
        self._schedule = schedule or []
        self.injuries: dict[int, Injury] = injuries or {}
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
        schedule_rows = []
        try:
            schedule_rows = cdn.schedule_rows()
            self._schedule = [ScheduledGame(**r) for r in schedule_rows]
        except Exception:
            log.exception("Could not load the schedule; next games must be entered by hand")

        frames, loaded = [], []
        try:
            frames.append(load_season(previous_season(season), settings=self.settings))
            loaded.append(previous_season(season))
        except SeasonUnavailableError:
            log.warning("No games for %s", previous_season(season))
        current = self._load_current(season, schedule_rows)
        if current is not None and not current.empty:
            frames.append(current)
            loaded.append(season)
        if not frames:
            raise SeasonUnavailableError("No league data available for the last two seasons")
        self._install(pd.concat(frames, ignore_index=True), loaded)
        self.refresh_injuries()

    def _load_current(self, season: str, schedule_rows: list[dict]) -> pd.DataFrame | None:
        """The season in progress: the local cache, topped up with new box scores from cdn.nba.com.

        cdn.nba.com works from cloud servers where stats.nba.com is blocked; stats.nba.com is the
        fallback when the CDN cannot be reached.
        """
        path = season_path(season, self.settings)
        cached = validate(pd.read_csv(path), str(path)) if path.exists() else None
        if schedule_rows:
            try:
                updated = cdn.update_season(cached, season, schedule_rows)
                if updated is not None and not updated.empty:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    updated.to_csv(path, index=False)
                return updated
            except Exception:
                log.exception("cdn.nba.com update failed, trying stats.nba.com")
        try:
            return load_season(season, max_age_hours=self.settings.live_refresh_hours, settings=self.settings)
        except Exception:
            log.exception("stats.nba.com failed too; serving the cached %s games", season)
            return cached

    def refresh_injuries(self) -> None:
        try:
            report = fetch_report()
            injuries = match(report, self.players, fold)
            with self._lock:
                self.injuries, self.injuries_at = injuries, time.time()
            log.info("Injury report: %d players matched", len(injuries))
        except Exception:
            log.exception("Injury report unavailable")

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
        roster = roster_snapshot(norm)
        with self._lock:
            self.raw, self.norm, self.players, self.snapshot, self.roster = raw, norm, players, snapshot, roster
            self.seasons = seasons or sorted(
                {f"{sid[1:]}-{str(int(sid[1:]) + 1)[-2:]}" for sid in norm["SEASON_ID"].unique()}
            )
            self.version += 1
            self.loaded_at = time.time()

    def _in_background(self, work: Callable[[], None]) -> None:
        if self._refreshing:
            return
        self._refreshing = True

        def run() -> None:
            try:
                work()
            except Exception:
                log.exception("Background refresh failed")
            finally:
                self._refreshing = False

        threading.Thread(target=run, daemon=True).start()

    def refresh_if_stale(self) -> None:
        if not self.loaded_at:
            return
        now = time.time()
        if now - self.loaded_at > self.settings.live_refresh_hours * 3600:
            self._in_background(self.load)
        elif self.injuries_at and now - self.injuries_at > self.settings.injury_refresh_minutes * 60:
            self._in_background(self.refresh_injuries)

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

    def regulars(self, team: str, on: date, exclude: int | None = None) -> pd.DataFrame:
        """The team's rotation going into a game, with each player's last 10 game form."""
        r = self.roster
        recent = r["GAME_DATE"] >= pd.Timestamp(on) - pd.Timedelta(days=REGULAR_DAYS)
        regs = r[(r["TEAM"] == team) & recent & (r["MIN"] >= REGULAR_MINUTES) & (r["PLAYER_ID"] != exclude)]
        return regs.sort_values("MIN", ascending=False)

    def injured_out(self, team: str, on: date, exclude: int | None = None) -> tuple[int, ...]:
        """Regulars the injury report lists as Out or Doubtful."""
        regs = self.regulars(team, on, exclude)
        return tuple(int(p) for p in regs["PLAYER_ID"] if (i := self.injuries.get(int(p))) and i.out)

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
