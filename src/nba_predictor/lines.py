"""Forward testing against real sportsbook lines (the-odds-api.com).

Historical player prop lines cost money, so the honest free test runs forward:

1. snapshot: before games, save each player prop's consensus line and best prices, plus the
   model's probability and pick, to a ledger.
2. grade: after games, fill in what happened and the profit of every pick at 1 unit.

Over weeks the ledger becomes the model's real record against real books. Off until
NBA_ODDS_API_KEY is set (free tier: 500 credits a month; each game and market costs one).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np
import pandas as pd

from .quant import american_to_decimal

if TYPE_CHECKING:
    from .config import Settings
    from .predictor import Predictor

log = logging.getLogger(__name__)

API = "https://api.the-odds-api.com/v4/sports/basketball_nba"
MARKET_TO_STAT = {"player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST", "player_threes": "FG3M"}
COLUMNS = [
    "snapshot_at",
    "game_date",
    "event_id",
    "player_id",
    "player",
    "stat",
    "line",
    "over_odds",
    "under_odds",
    "books",
    "p_over",
    "p_under",
    "pick",
    "pick_odds",
    "edge",
    "ev",
    "actual",
    "result",
    "profit",
]


def consensus(outcomes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Median line across books, and the best over and under price offered at that line."""
    lines = [o["point"] for o in outcomes if o.get("point") is not None]
    if not lines:
        return None
    line = median(lines)
    if line % 0.5:  # a median between two lines: snap to the most common one
        line = max(set(lines), key=lines.count)
    at_line = [o for o in outcomes if o.get("point") == line]
    overs = [o["price"] for o in at_line if o["name"] == "Over"]
    unders = [o["price"] for o in at_line if o["name"] == "Under"]
    if not overs or not unders:
        return None
    best = lambda prices: max(prices, key=american_to_decimal)  # noqa: E731
    return {
        "line": float(line),
        "over_odds": best(overs),
        "under_odds": best(unders),
        "books": len({o["book"] for o in at_line}),
    }


def parse_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (player, market) from an event odds response."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            for o in market.get("outcomes", []):
                key = (o.get("description", ""), market["key"])
                grouped.setdefault(key, []).append({**o, "book": book["key"]})
    rows = []
    for (player, market), outcomes in grouped.items():
        c = consensus(outcomes)
        if c and market in MARKET_TO_STAT:
            rows.append(
                {
                    "player": player,
                    "stat": MARKET_TO_STAT[market],
                    "event_id": event["id"],
                    "commence_time": event["commence_time"],
                    **c,
                }
            )
    return rows


def fetch_props(key: str, markets: list[str], within_hours: float = 30) -> list[dict[str, Any]]:
    with httpx.Client(timeout=30) as client:
        events = client.get(f"{API}/events", params={"apiKey": key}).raise_for_status().json()
        soon = datetime.now(UTC) + timedelta(hours=within_hours)
        rows: list[dict[str, Any]] = []
        for e in events:
            if pd.Timestamp(e["commence_time"]) > pd.Timestamp(soon):
                continue
            r = client.get(
                f"{API}/events/{e['id']}/odds",
                params={
                    "apiKey": key,
                    "regions": "us",
                    "markets": ",".join(markets),
                    "oddsFormat": "american",
                },
            )
            r.raise_for_status()
            log.info("Odds API credits remaining: %s", r.headers.get("x-requests-remaining"))
            rows += parse_event(r.json())
    return rows


def read_ledger(settings: Settings) -> pd.DataFrame:
    path = settings.ledger_path
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)


def snapshot(predictor: Predictor, props: list[dict[str, Any]]) -> pd.DataFrame:
    """Price every prop with the model and add (or refresh) it in the ledger."""
    from .store import fold

    store = predictor.store
    names = store.players["name"]
    by_name = {fold(n): pid for pid, n in zip(names.index.astype(int).tolist(), names.tolist(), strict=True)}
    now = datetime.now(UTC).isoformat(timespec="seconds")
    new = []
    for p in props:
        pid = by_name.get(fold(p["player"]))
        if pid is None:
            continue
        try:
            proj = predictor.project([{"player_id": pid}])[0]
        except (KeyError, LookupError):
            continue
        q = predictor.price(proj, p["stat"], p["line"], p["over_odds"], p["under_odds"], 1.0, 0.25)
        side = q[q["recommendation"]] if q["recommendation"] != "pass" else None
        new.append(
            {
                "snapshot_at": now,
                "game_date": proj.context.game_date.isoformat(),
                "event_id": p["event_id"],
                "player_id": pid,
                "player": proj.name,
                "stat": p["stat"],
                "line": p["line"],
                "over_odds": p["over_odds"],
                "under_odds": p["under_odds"],
                "books": p["books"],
                "p_over": round(q["p_over"], 4),
                "p_under": round(q["p_under"], 4),
                "pick": q["recommendation"],
                "pick_odds": side["odds"] if side else None,
                "edge": round(side["edge"], 4) if side else None,
                "ev": round(side["expected_value"], 4) if side else None,
                "actual": None,
                "result": None,
                "profit": None,
            }
        )
    ledger = read_ledger(predictor.settings)
    fresh = pd.DataFrame(new, columns=COLUMNS)
    if len(ledger) and len(fresh):  # replace earlier snapshots of the same ungraded prop with the latest
        key = ["game_date", "player_id", "stat"]
        stale = ledger.set_index(key).index.isin(fresh.set_index(key).index) & ledger["result"].isna()
        ledger = ledger[~stale]
    out = pd.concat([ledger, fresh], ignore_index=True) if len(ledger) else fresh
    predictor.settings.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(predictor.settings.ledger_path, index=False)
    return out


def grade(ledger: pd.DataFrame, game_logs: pd.DataFrame) -> pd.DataFrame:
    """Fill in results for props whose game is in the logs. Picks are 1 unit; passes have no profit."""
    dates = pd.to_datetime(game_logs["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    played = {
        (int(p), d): row for p, d, row in zip(game_logs["PLAYER_ID"], dates, game_logs.to_dict("records"), strict=True)
    }
    ledger = ledger.copy()
    for i, row in ledger[ledger["result"].isna()].iterrows():
        box = played.get((int(row["player_id"]), str(row["game_date"])))
        if box is None:
            continue
        actual = float(box[row["stat"]])
        outcome = "push" if actual == row["line"] else ("over" if actual > row["line"] else "under")
        profit = None
        if row["pick"] in ("over", "under"):
            if outcome == "push":
                profit = 0.0
            elif outcome == row["pick"]:
                profit = american_to_decimal(float(row["pick_odds"])) - 1
            else:
                profit = -1.0
        ledger.loc[i, ["actual", "result", "profit"]] = [actual, outcome, profit]
    return ledger


def summary(ledger: pd.DataFrame) -> dict[str, Any]:
    graded = ledger[ledger["result"].notna()]
    picks = graded[graded["pick"].isin(["over", "under"])]
    decided = picks[picks["result"] != "push"]
    wins = int((decided["pick"] == decided["result"]).sum())
    went_over = (graded["result"] == "over").astype(float)
    return {
        "props_tracked": len(ledger),
        "graded": len(graded),
        "picks": len(picks),
        "wins": wins,
        "losses": int(len(decided) - wins),
        "pushes": int((picks["result"] == "push").sum()),
        "units": round(float(picks["profit"].sum()), 2) if len(picks) else 0.0,
        "roi": round(float(picks["profit"].sum() / len(picks)), 4) if len(picks) else None,
        "brier_model": round(float(np.mean((graded["p_over"] - went_over) ** 2)), 4) if len(graded) else None,
        "pending": int(ledger["result"].isna().sum()),
    }
