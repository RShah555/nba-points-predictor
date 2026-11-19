"""Serving logic: projections, simulations, prop pricing, start/sit and the daily slate."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd
from cachetools import TTLCache

from . import distributions as dist
from .config import Settings, get_settings
from .fantasy import PRESETS, ScoringSystem, doubles, score
from .features import TARGETS, GameContext, next_game_features
from .modeling import ModelBundle, explain
from .quant import price_prop
from .seasons import current_season, season_id
from .store import LeagueStore

COMBO_MARKETS: dict[str, tuple[str, ...]] = {
    "PRA": ("PTS", "REB", "AST"),
    "PR": ("PTS", "REB"),
    "PA": ("PTS", "AST"),
    "RA": ("REB", "AST"),
    "STOCKS": ("STL", "BLK"),
}
MARKET_LABELS = {
    "PTS": "Points",
    "REB": "Rebounds",
    "AST": "Assists",
    "FG3M": "3 pointers made",
    "STL": "Steals",
    "BLK": "Blocks",
    "TOV": "Turnovers",
    "FGM": "Field goals made",
    "FGA": "Field goal attempts",
    "FTM": "Free throws made",
    "FTA": "Free throw attempts",
    "PRA": "Pts + Reb + Ast",
    "PR": "Pts + Reb",
    "PA": "Pts + Ast",
    "RA": "Reb + Ast",
    "STOCKS": "Steals + Blocks",
    **{f"FPTS_{k}": f"{p.name} fantasy points" for k, p in PRESETS.items()},
}
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


@dataclass
class Projection:
    player_id: int
    name: str
    team: str
    context: GameContext
    context_source: str  # "schedule" or "manual"
    rest_days: int
    features: pd.DataFrame
    means: dict[str, float]
    sizes: dict[str, float]
    _samples: np.ndarray | None = field(default=None, repr=False)


class Predictor:
    def __init__(self, bundle: ModelBundle, store: LeagueStore, settings: Settings | None = None):
        self.bundle = bundle
        self.store = store
        self.settings = settings or get_settings()
        self._cache: TTLCache[tuple, Projection] = TTLCache(maxsize=2048, ttl=900)

    # contexts ------------------------------------------------------------------
    def context(
        self,
        player_id: int,
        opponent: str | None = None,
        home: bool | None = None,
        rest_days: int | None = None,
        teammates_out: list[int] | None = None,
    ) -> tuple[GameContext, str, int]:
        """The next game for a player. Anything not given comes from the schedule and injury report."""
        info = self.store.player(player_id)
        last_game: date = pd.Timestamp(info["last_game"]).date()
        team = str(info["team"])
        source = "manual"
        game_date = None
        if opponent is None:
            nxt = self.store.next_game(team, after=max(self.store.today, last_game + timedelta(days=1)))
            if nxt is None:
                raise LookupError(f"No upcoming game found for {team}; choose an opponent")
            opponent = nxt.away if nxt.home == team else nxt.home
            home = nxt.home == team if home is None else home
            game_date, source = nxt.game_date, "schedule"
        if rest_days is not None:
            game_date = last_game + timedelta(days=rest_days)
        if game_date is None:
            game_date = max(self.store.today, last_game + timedelta(days=1))
        rest = (game_date - last_game).days
        if teammates_out is None:
            out = self.store.injured_out(team, game_date, exclude=player_id)
        else:
            out = tuple(sorted(set(teammates_out) - {player_id}))
        ctx = GameContext(
            player_id,
            opponent.upper(),
            bool(home if home is not None else True),
            game_date,
            season_id(current_season(game_date)),
            team,
            out,
        )
        return ctx, source, rest

    # projections -----------------------------------------------------------------
    def project(self, requests: list[dict[str, Any]]) -> list[Projection]:
        """Batch projections. Each request: player_id and optional opponent, home, rest_days."""
        self.store.refresh_if_stale()
        out: list[Projection | None] = [None] * len(requests)
        todo: list[tuple[int, GameContext, str, int, tuple]] = []
        for i, r in enumerate(requests):
            ctx, source, rest = self.context(
                r["player_id"], r.get("opponent"), r.get("home"), r.get("rest_days"), r.get("teammates_out")
            )
            key = (self.store.version, ctx, source)
            if key in self._cache:
                out[i] = self._cache[key]
            else:
                todo.append((i, ctx, source, rest, key))
        if todo:
            contexts = [t[1] for t in todo]
            feats = next_game_features(
                self.store.history([c.player_id for c in contexts]), contexts, self.store.snapshot, self.store.roster
            )
            means = self.bundle.predict_means(feats)
            sizes = self.bundle.sizes(means)
            for row, (i, ctx, source, rest, key) in enumerate(todo):
                info = self.store.player(ctx.player_id)
                proj = Projection(
                    ctx.player_id,
                    str(info["name"]),
                    str(info["team"]),
                    ctx,
                    source,
                    rest,
                    feats.iloc[[row]].reset_index(drop=True),
                    {t: float(means.iloc[row][t]) for t in TARGETS},
                    {t: float(sizes.iloc[row][t]) for t in TARGETS},
                )
                self._cache[key] = proj
                out[i] = proj
        return [p for p in out if p is not None]

    def samples(self, proj: Projection, n: int | None = None) -> np.ndarray:
        n = n or self.settings.simulations
        if proj._samples is None or len(proj._samples) != n:
            seed = int(hashlib.sha256(repr(proj.context).encode()).hexdigest()[:8], 16)
            mu = np.array([proj.means[t] for t in self.bundle.targets])
            k = np.array([proj.sizes[t] for t in self.bundle.targets])
            proj._samples = dist.sample_joint(
                mu, k, self.bundle.copula, n, np.random.default_rng(seed), self.bundle.recalibrations
            )
        return proj._samples

    def sample_dict(self, proj: Projection, n: int | None = None) -> dict[str, np.ndarray]:
        s = self.samples(proj, n)
        return {t: s[:, j] for j, t in enumerate(self.bundle.targets)}

    def market_samples(self, proj: Projection, market: str, scoring: ScoringSystem | None = None) -> np.ndarray:
        draws = self.sample_dict(proj)
        if market in COMBO_MARKETS:
            return np.sum([draws[s] for s in COMBO_MARKETS[market]], axis=0)
        if market.startswith("FPTS_"):
            return score(draws, scoring or PRESETS[market.removeprefix("FPTS_")])
        return draws[market]

    # payloads ------------------------------------------------------------------
    def stat_line(self, proj: Projection, stat: str) -> dict[str, Any]:
        mu, k = proj.means[stat], proj.sizes[stat]
        recal = self.bundle.models[stat].recalibration
        q = dist.ppf(np.array(QUANTILES), mu, k, recal)
        support, pmf = dist.pmf_table(mu, k, recal)
        f = proj.features.iloc[0]
        return {
            "stat": stat,
            "label": MARKET_LABELS[stat],
            "mean": round(mu, 2),
            "median": float(q[2]),
            "quantiles": {f"p{int(p * 100)}": float(v) for p, v in zip(QUANTILES, q, strict=True)},
            "sd": round(float(np.sqrt(mu + mu**2 / k)), 2),
            "season_avg": _num(f.get(f"SEASON_AVG_{stat}")),
            "last_5": _num(f.get(f"{stat}_roll_5")),
            "last_10": _num(f.get(f"{stat}_roll_10")),
            "distribution": [{"value": int(v), "p": round(float(p), 5)} for v, p in zip(support, pmf, strict=True)],
        }

    def fantasy_summary(self, proj: Projection, system: ScoringSystem) -> dict[str, Any]:
        pts = score(self.sample_dict(proj), system)
        hist, edges = np.histogram(pts, bins=30)
        return {
            "key": system.key,
            "name": system.name,
            "mean": round(float(pts.mean()), 2),
            "quantiles": {f"p{int(p * 100)}": round(float(np.quantile(pts, p)), 1) for p in QUANTILES},
            "histogram": [
                {"x": round(float((edges[i] + edges[i + 1]) / 2), 1), "p": round(float(h / len(pts)), 4)}
                for i, h in enumerate(hist)
            ],
        }

    def full_payload(self, proj: Projection, explain_stats: tuple[str, ...] = TARGETS) -> dict[str, Any]:
        draws = self.sample_dict(proj)
        n_doubles = doubles(draws)
        return {
            "player": self.player_payload(proj.player_id),
            "game": self.context_payload(proj),
            "stats": {t: self.stat_line(proj, t) for t in self.bundle.targets},
            "combos": {m: _summary(self.market_samples(proj, m)) for m in COMBO_MARKETS},
            "fantasy": {k: self.fantasy_summary(proj, p) for k, p in PRESETS.items()},
            "milestones": {
                "double_double": round(float(np.mean(n_doubles >= 2)), 4),
                "triple_double": round(float(np.mean(n_doubles >= 3)), 4),
                "pts_20_plus": round(float(np.mean(draws["PTS"] >= 20)), 4),
                "pts_30_plus": round(float(np.mean(draws["PTS"] >= 30)), 4),
                "fg3m_3_plus": round(float(np.mean(draws["FG3M"] >= 3)), 4),
            },
            "explanations": {t: explain(self.bundle.models[t], proj.features) for t in explain_stats},
        }

    def player_payload(self, player_id: int) -> dict[str, Any]:
        info = self.store.player(player_id)
        return {
            "id": int(player_id),
            "name": str(info["name"]),
            "team": str(info["team"]),
            "team_id": _int(info.get("team_id")),
            "games": int(info["games"]),
            "pts": _num(info["pts"]),
            "reb": _num(info["reb"]),
            "ast": _num(info["ast"]),
            "min": _num(info["min"]),
            "headshot": f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(player_id)}.png",
            "injury": self.injury_payload(player_id),
        }

    def injury_payload(self, player_id: int) -> dict[str, Any] | None:
        i = self.store.injuries.get(int(player_id))
        if i is None:
            return None
        return {"status": i.status, "note": i.note, "return_date": i.return_date, "out": i.out}

    def context_payload(self, proj: Projection, teammates: bool = True) -> dict[str, Any]:
        c = proj.context
        base = {
            "opponent": c.opponent,
            "home": c.home,
            "date": c.game_date.isoformat(),
            "rest_days": proj.rest_days,
            "back_to_back": proj.rest_days == 1,
            "source": proj.context_source,
            "teammates_out": len(c.teammates_out),
        }
        if not teammates:
            return base
        regulars = self.store.regulars(c.team or proj.team, c.game_date, exclude=proj.player_id)
        return {
            **base,
            "teammates": [
                {
                    "id": pid,
                    "name": str(self.store.player(pid)["name"]),
                    "min": round(m, 1),
                    "pts": round(pts, 1),
                    "injury": self.injury_payload(pid),
                    "out": pid in c.teammates_out,
                }
                for pid, m, pts in zip(
                    regulars["PLAYER_ID"].astype(int).tolist(),
                    regulars["MIN"].astype(float).tolist(),
                    regulars["PTS"].astype(float).tolist(),
                    strict=True,
                )
            ],
        }

    # props ---------------------------------------------------------------------
    def price(
        self,
        proj: Projection,
        market: str,
        line: float,
        over_odds: float,
        under_odds: float,
        bankroll: float,
        kelly_multiplier: float,
        max_fraction: float = 0.05,
    ) -> dict[str, Any]:
        if market in self.bundle.targets:
            p_over, p_under, p_push = dist.over_under(
                line, proj.means[market], proj.sizes[market], self.bundle.models[market].recalibration
            )
            method = "calibrated negative binomial"
            draws = self.market_samples(proj, market)
        else:
            draws = self.market_samples(proj, market)
            p_over, p_push = float(np.mean(draws > line)), float(np.mean(draws == line))
            p_under = 1 - p_over - p_push
            method = f"{len(draws):,} copula simulations"
        quote = price_prop(p_over, p_under, p_push, over_odds, under_odds, bankroll, kelly_multiplier, max_fraction)
        return {
            "market": market,
            "label": MARKET_LABELS[market],
            "line": line,
            "method": method,
            "projection": _summary(draws),
            **quote.to_dict(),
        }

    # start / sit -----------------------------------------------------------------
    def compare(self, projs: list[Projection], system: ScoringSystem) -> dict[str, Any]:
        sims = [score(self.sample_dict(p), system) for p in projs]
        n = len(projs)
        beats = [
            [
                None if i == j else round(float(np.mean(sims[i] > sims[j]) + 0.5 * np.mean(sims[i] == sims[j])), 4)
                for j in range(n)
            ]
            for i in range(n)
        ]
        best = np.argmax(np.stack(sims), axis=0)
        return {
            "scoring": system.name,
            "players": [
                {
                    "player": self.player_payload(p.player_id),
                    "game": self.context_payload(p, teammates=False),
                    "fantasy": self.fantasy_summary(p, system),
                    "p_best": round(float(np.mean(best == i)), 4),
                }
                for i, p in enumerate(projs)
            ],
            "p_beats": beats,
            "note": "Players are simulated independently; teammates' outcomes are slightly correlated in reality.",
        }

    # slate -----------------------------------------------------------------------
    def slate(self, min_minutes: float = 18.0, on: date | None = None) -> dict[str, Any]:
        day, games = self.store.slate(on)
        if not games:
            return {"date": None, "games": [], "players": []}
        assert day is not None
        open_games = [g for g in games if g.status != "final"]  # finished games are shown, not projected
        teams = {g.home for g in open_games} | {g.away for g in open_games}
        table = self.store.players
        cutoff = pd.Timestamp(self.store.today - timedelta(days=21))
        ruled_out = [pid for pid, i in self.store.injuries.items() if i.out]
        pool = table[
            table["team"].isin(teams)
            & (table["min_last_10"] >= min_minutes)
            & (pd.to_datetime(table["last_game"]) >= cutoff)
            & ~table.index.isin(ruled_out)
        ]
        requests = []
        for pid, row in pool.iterrows():
            g = next(g for g in open_games if row["team"] in (g.home, g.away))
            requests.append(
                {
                    "player_id": int(cast(int, pid)),
                    "opponent": g.away if g.home == row["team"] else g.home,
                    "home": g.home == row["team"],
                    "rest_days": (day - pd.Timestamp(row["last_game"]).date()).days,
                }
            )
        projs = self.project(requests) if requests else []
        players = []
        for p in projs:
            draws = self.sample_dict(p, 2000)
            players.append(
                {
                    "player": self.player_payload(p.player_id),
                    "game": self.context_payload(p, teammates=False),
                    "mean": {t: round(v, 2) for t, v in p.means.items()},
                    "fantasy": {k: round(float(score(draws, s).mean()), 2) for k, s in PRESETS.items()},
                }
            )
        players.sort(key=lambda x: -x["fantasy"]["espn"])
        return {
            "date": day.isoformat() if day else None,
            "games": [
                {"id": g.game_id, "home": g.home, "away": g.away, "status": g.status, "tipoff_utc": g.tipoff_utc}
                for g in games
            ],
            "players": players,
        }


def _summary(draws: np.ndarray) -> dict[str, Any]:
    return {
        "mean": round(float(np.mean(draws)), 2),
        **{f"p{int(p * 100)}": round(float(np.quantile(draws, p)), 1) for p in QUANTILES},
    }


def _num(v: Any) -> float | None:
    return None if v is None or pd.isna(v) else round(float(v), 2)


def _int(v: Any) -> int | None:
    return None if v is None or pd.isna(v) else int(v)
