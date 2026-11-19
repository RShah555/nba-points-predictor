"""FastAPI service.

uvicorn nba_predictor.api:create_app --factory --reload
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .. import __version__
from ..config import Settings, get_settings
from ..fantasy import PRESETS
from ..predictor import MARKET_LABELS, Predictor
from ..store import LeagueStore
from .schemas import MARKETS, CompareRequest, GameRequest, PropRequest

log = logging.getLogger("nba_predictor.api")


def create_app(predictor: Predictor | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    state: dict[str, Any] = {"predictor": predictor, "error": None, "report": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if state["predictor"] is None:
            try:
                from ..modeling import ModelBundle

                bundle = ModelBundle.load(settings.bundle_path)
                store = LeagueStore(settings)
                await run_in_threadpool(store.load)
                state["predictor"] = Predictor(bundle, store, settings)
            except FileNotFoundError:
                state["error"] = "No trained model found. Run `nba-predictor train` first."
            except Exception as e:  # keep the API up so /health can say what is wrong
                log.exception("Startup failed")
                state["error"] = f"Startup failed: {e}"
        if settings.report_path.exists():
            state["report"] = json.loads(settings.report_path.read_text())
        yield

    app = FastAPI(
        title="NBA Predictor API",
        version=__version__,
        lifespan=lifespan,
        description="Probabilistic next game projections, fantasy points and prop pricing.",
    )
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["server-timing"] = f"app;dur={elapsed:.1f}"
        log.info("%s %s %s %.0fms id=%s", request.method, request.url.path, response.status_code, elapsed, request_id)
        return response

    @app.exception_handler(KeyError)
    async def not_found(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc).strip("'\"")})

    @app.exception_handler(LookupError)
    async def no_game(_: Request, exc: LookupError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    def ready() -> Predictor:
        if state["predictor"] is None:
            raise HTTPException(503, state["error"] or "Model is loading")
        return state["predictor"]

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        p = state["predictor"]
        return {
            "status": "ok" if p else "degraded",
            "version": __version__,
            "error": state["error"],
            "data_version": p.store.version if p else None,
        }

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        p = ready()
        return {
            "version": __version__,
            "model_trained_at": p.bundle.trained_at,
            "season": p.store.season,
            "data_seasons": p.store.seasons,
            "data_loaded_at": p.store.loaded_at,
            "today": p.store.today.isoformat(),
            "targets": p.bundle.targets,
            "markets": [{"key": m, "label": MARKET_LABELS[m]} for m in MARKETS],
            "scoring": [
                {"key": k, "name": s.name, "weights": dict(s.weights), "bonuses": [b.__dict__ for b in s.bonuses]}
                for k, s in PRESETS.items()
            ],
            "teams": sorted(p.store.snapshot.index.tolist()),
            "headline": p.bundle.metrics.get("headline"),
        }

    @app.get("/api/players")
    def players(q: str = "", limit: int = Query(20, ge=1, le=100)) -> list[dict[str, Any]]:
        p = ready()
        return [p.player_payload(int(pid)) for pid in p.store.search(q, limit).index]

    @app.get("/api/players/{player_id}")
    def player(player_id: int) -> dict[str, Any]:
        p = ready()
        log_df = p.store.game_log(player_id)
        if log_df.empty:
            raise KeyError(f"Unknown player {player_id}")
        games = log_df.tail(25).iloc[::-1]
        cols = [
            "GAME_DATE",
            "MATCHUP",
            "WL",
            "MIN",
            "PTS",
            "REB",
            "AST",
            "FG3M",
            "STL",
            "BLK",
            "TOV",
            "FGM",
            "FGA",
            "PLUS_MINUS",
        ]
        rows = games[[c for c in cols if c in games]].assign(GAME_DATE=games["GAME_DATE"].dt.strftime("%Y-%m-%d"))
        return {"player": p.player_payload(player_id), "games": json.loads(rows.to_json(orient="records"))}

    @app.post("/api/projections")
    async def projection(req: GameRequest) -> dict[str, Any]:
        p = ready()

        def work() -> dict[str, Any]:
            proj = p.project([req.model_dump()])[0]
            return p.full_payload(proj)

        return await run_in_threadpool(work)

    @app.post("/api/props")
    async def prop(req: PropRequest) -> dict[str, Any]:
        p = ready()

        def work() -> dict[str, Any]:
            proj = p.project([req.model_dump(include={"player_id", "opponent", "home", "rest_days"})])[0]
            return {
                "player": p.player_payload(proj.player_id),
                "game": p.context_payload(proj),
                **p.price(
                    proj,
                    req.market,
                    req.line,
                    req.over_odds,
                    req.under_odds,
                    req.bankroll,
                    req.kelly_multiplier,
                    req.max_fraction,
                ),
            }

        return await run_in_threadpool(work)

    @app.post("/api/compare")
    async def compare(req: CompareRequest) -> dict[str, Any]:
        p = ready()

        def work() -> dict[str, Any]:
            projs = p.project([r.model_dump() for r in req.players])
            return p.compare(projs, PRESETS[req.scoring])

        return await run_in_threadpool(work)

    @app.get("/api/slate")
    async def slate(min_minutes: float = Query(18, ge=0, le=48)) -> dict[str, Any]:
        p = ready()
        return await run_in_threadpool(p.slate, min_minutes)

    @app.get("/api/model/report")
    def report() -> dict[str, Any]:
        if state["report"] is None:
            raise HTTPException(404, "No training report found. Run `nba-predictor train` first.")
        return state["report"]

    return app
