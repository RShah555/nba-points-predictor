"""Request models. Responses are plain JSON built by the Predictor."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..fantasy import PRESETS
from ..features import TARGETS
from ..predictor import COMBO_MARKETS

MARKETS = (*TARGETS, *COMBO_MARKETS, *(f"FPTS_{k}" for k in PRESETS))


class GameRequest(BaseModel):
    player_id: int
    opponent: str | None = Field(None, description="Team abbreviation; omitted means the next scheduled game")
    home: bool | None = None
    rest_days: int | None = Field(None, ge=1, le=30)

    @field_validator("opponent")
    @classmethod
    def upper(cls, v: str | None) -> str | None:
        return v.upper() if v else v


class PropRequest(GameRequest):
    market: str = "PTS"
    line: float = Field(..., ge=0, le=200)
    over_odds: float = -110
    under_odds: float = -110
    bankroll: float = Field(1000, gt=0)
    kelly_multiplier: float = Field(0.25, gt=0, le=1)
    max_fraction: float = Field(0.05, gt=0, le=1)

    @field_validator("market")
    @classmethod
    def known_market(cls, v: str) -> str:
        v = v.strip()
        v = "FPTS_" + v[5:].lower() if v.upper().startswith("FPTS_") else v.upper()
        if v not in MARKETS:
            raise ValueError(f"market must be one of {', '.join(MARKETS)}")
        return v

    @field_validator("over_odds", "under_odds")
    @classmethod
    def american(cls, v: float) -> float:
        if -100 < v < 100:
            raise ValueError("American odds must be <= -100 or >= +100")
        return v


class CompareRequest(BaseModel):
    players: list[GameRequest] = Field(..., min_length=2, max_length=6)
    scoring: Literal["espn", "yahoo", "sleeper"] = "espn"
