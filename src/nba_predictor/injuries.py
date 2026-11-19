"""The league injury report, from ESPN's public feed, matched to NBA.com player ids."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import pandas as pd

log = logging.getLogger(__name__)

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
# ESPN abbreviations that differ from NBA.com tricodes
ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX"}


@dataclass(frozen=True)
class Injury:
    player_id: int
    name: str
    team: str
    status: str  # "Out", "Doubtful", "Questionable", "Day-To-Day", ...
    note: str
    return_date: str | None

    @property
    def out(self) -> bool:
        """Treated as not playing: listed Out or Doubtful."""
        return self.status.lower() in {"out", "doubtful"}


def fetch_report() -> list[dict]:
    r = httpx.get(URL, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=30)
    r.raise_for_status()
    rows = []
    for team in r.json().get("injuries", []):
        for i in team.get("injuries", []):
            athlete = i.get("athlete", {})
            abbr = (athlete.get("team") or {}).get("abbreviation", "")
            details = i.get("details") or {}
            rows.append(
                {
                    "name": athlete.get("displayName", ""),
                    "team": ESPN_TO_NBA.get(abbr, abbr),
                    "status": i.get("status", ""),
                    "note": i.get("shortComment", ""),
                    "return_date": details.get("returnDate"),
                }
            )
    return rows


def match(report: list[dict], players: pd.DataFrame, fold: Callable[[str], str]) -> dict[int, Injury]:
    """Attach NBA.com ids by accent-insensitive name, preferring a match on the same team."""
    by_name: dict[str, list[tuple[int, str]]] = {}
    for pid, name, team in zip(players.index.astype(int).tolist(), players["name"], players["team"], strict=True):
        by_name.setdefault(fold(name), []).append((pid, str(team)))
    out: dict[int, Injury] = {}
    for r in report:
        candidates = by_name.get(fold(r["name"]), [])
        same_team = [c for c in candidates if c[1] == r["team"]]
        if not (same_team or candidates):
            continue
        pick = (same_team or candidates)[0]
        out[pick[0]] = Injury(pick[0], r["name"], r["team"], r["status"], r["note"], r["return_date"])
    return out
