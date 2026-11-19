"""Fantasy scoring for points leagues.

The presets follow each platform's default head to head points scoring at the
time of writing. Leagues customise these, so any ScoringSystem can be passed in.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

DD_STATS = ("PTS", "REB", "AST", "STL", "BLK")


@dataclass(frozen=True)
class Bonus:
    kind: Literal["threshold", "double_double", "triple_double"]
    points: float
    stat: str | None = None
    value: float = 0.0


@dataclass(frozen=True)
class ScoringSystem:
    key: str
    name: str
    weights: Mapping[str, float]
    bonuses: tuple[Bonus, ...] = field(default_factory=tuple)

    @property
    def stats_needed(self) -> set[str]:
        needed = set(self.weights)
        for b in self.bonuses:
            needed |= set(DD_STATS) if b.kind != "threshold" else {b.stat or ""}
        return needed - {""}


PRESETS: dict[str, ScoringSystem] = {
    "espn": ScoringSystem(
        "espn",
        "ESPN",
        {
            "PTS": 1,
            "FGM": 2,
            "FGA": -1,
            "FTM": 1,
            "FTA": -1,
            "FG3M": 1,
            "REB": 1,
            "AST": 2,
            "STL": 4,
            "BLK": 4,
            "TOV": -2,
        },
    ),
    "yahoo": ScoringSystem(
        "yahoo",
        "Yahoo",
        {"PTS": 1, "REB": 1.2, "AST": 1.5, "STL": 3, "BLK": 3, "TOV": -1},
    ),
    "sleeper": ScoringSystem(
        "sleeper",
        "Sleeper",
        {"PTS": 0.5, "FG3M": 0.5, "REB": 1, "AST": 1, "STL": 2, "BLK": 2, "TOV": -1},
        (
            Bonus("double_double", 1),
            Bonus("triple_double", 2),
            Bonus("threshold", 2, "PTS", 40),
            Bonus("threshold", 2, "PTS", 50),
            Bonus("threshold", 2, "REB", 20),
            Bonus("threshold", 2, "AST", 15),
        ),
    ),
}


def doubles(stats: Mapping[str, np.ndarray]) -> np.ndarray:
    """How many of PTS, REB, AST, STL, BLK reached double figures, per draw."""
    return np.sum([np.asarray(stats[s]) >= 10 for s in DD_STATS], axis=0)


def score(stats: Mapping[str, np.ndarray], system: ScoringSystem) -> np.ndarray:
    first = next(iter(stats.values()))
    total = np.zeros(np.shape(first), dtype=float)
    for stat, weight in system.weights.items():
        total += weight * np.asarray(stats[stat], dtype=float)
    if system.bonuses:
        n_doubles = doubles(stats)
        for b in system.bonuses:
            if b.kind == "threshold":
                total += b.points * (np.asarray(stats[b.stat or ""]) >= b.value)
            elif b.kind == "double_double":
                total += b.points * (n_doubles >= 2)
            else:
                total += b.points * (n_doubles >= 3)
    return total
