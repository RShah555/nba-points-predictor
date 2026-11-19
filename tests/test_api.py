"""End to end: train a small model on synthetic data, serve it, and call every endpoint."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nba_predictor import distributions as dist
from nba_predictor.api import create_app
from nba_predictor.config import Settings
from nba_predictor.features import TARGETS, build_training_frame
from nba_predictor.modeling import DEFAULT_PARAMS, ModelBundle, TargetModel, fit_booster
from nba_predictor.predictor import Predictor
from nba_predictor.store import LeagueStore, ScheduledGame


@pytest.fixture(scope="module")
def client(league, tmp_path_factory) -> TestClient:
    frame, _ = build_training_frame(league)
    rng = np.random.default_rng(0)
    models = {}
    for t in TARGETS:
        booster = fit_booster({**DEFAULT_PARAMS, "seed": 0, "min_data_in_leaf": 20}, frame, frame[t], 40)
        mu = booster.predict(frame[booster.feature_name()].astype(float))
        models[t] = TargetModel.from_booster(t, booster, dist.fit_stat_model(frame[t].to_numpy(), mu, rng), {})
    bundle = ModelBundle(
        "test", "2026-01-01T00:00:00", models, np.eye(len(TARGETS)), ["2024-25", "2025-26"], {"headline": {}}
    )

    last = pd.to_datetime(league["GAME_DATE"]).max().date()
    today = last + timedelta(days=1)
    schedule = [
        ScheduledGame("1", today, "BOS", "NYK", "scheduled", None),
        ScheduledGame("2", today, "LAL", "GSW", "scheduled", None),
    ]
    settings = Settings(artifacts_dir=tmp_path_factory.mktemp("models"), live_refresh_hours=1e6)
    store = LeagueStore(settings, raw=league, schedule=schedule, today=today)
    return TestClient(create_app(Predictor(bundle, store, settings), settings))


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"


def test_meta_lists_markets_and_scoring(client):
    body = client.get("/api/meta").json()
    keys = {m["key"] for m in body["markets"]}
    assert {"PTS", "PRA", "FPTS_espn"} <= keys
    assert {s["key"] for s in body["scoring"]} == {"espn", "yahoo", "sleeper"}


def test_player_search_and_detail(client):
    hits = client.get("/api/players", params={"q": "bos player"}).json()
    assert hits and all(h["team"] == "BOS" for h in hits)
    detail = client.get(f"/api/players/{hits[0]['id']}").json()
    assert detail["games"] and "PTS" in detail["games"][0]


def test_unfinished_games_are_dropped_from_history(league):
    """LeagueGameLog lists live games with partial box scores; those rows must never reach the features."""
    last_day = pd.to_datetime(league["GAME_DATE"]).max().date()
    live_ids = league.loc[pd.to_datetime(league["GAME_DATE"]).dt.date == last_day, "GAME_ID"].unique()
    schedule = [ScheduledGame(str(g), last_day, "BOS", "NYK", "live", None) for g in live_ids]
    store = LeagueStore(Settings(), raw=league, schedule=schedule, today=last_day)
    assert pd.to_datetime(store.raw["GAME_DATE"]).max().date() < last_day
    no_schedule = LeagueStore(Settings(), raw=league, today=last_day)
    assert pd.to_datetime(no_schedule.raw["GAME_DATE"]).max().date() < last_day


def test_search_ignores_accents():
    from nba_predictor.store import fold

    assert fold("Nikola Jokić") == "nikola jokic"
    assert fold("Luka Dončić") == "luka doncic"
    assert "jokic" in fold("NIKOLA JOKIĆ ")


def test_projection_uses_the_schedule(client):
    body = client.post("/api/projections", json={"player_id": 0}).json()
    assert body["game"] == {**body["game"], "opponent": "NYK", "home": True, "source": "schedule"}
    pts = body["stats"]["PTS"]
    assert pts["quantiles"]["p10"] <= pts["median"] <= pts["quantiles"]["p90"]
    assert sum(d["p"] for d in pts["distribution"]) == pytest.approx(1, abs=0.01)
    steps = body["explanations"]["PTS"]["steps"]
    assert body["explanations"]["PTS"]["base"] + sum(s["delta"] for s in steps) == pytest.approx(pts["mean"], abs=0.01)
    assert set(body["fantasy"]) == {"espn", "yahoo", "sleeper"}


def test_manual_context(client):
    body = client.post(
        "/api/projections", json={"player_id": 0, "opponent": "lal", "home": False, "rest_days": 1}
    ).json()
    assert body["game"]["opponent"] == "LAL" and body["game"]["back_to_back"]


def test_props(client):
    body = client.post("/api/props", json={"player_id": 0, "market": "PTS", "line": 15.5}).json()
    assert body["p_over"] + body["p_under"] + body["p_push"] == pytest.approx(1)
    assert body["recommendation"] in {"over", "under", "pass"}
    combo = client.post("/api/props", json={"player_id": 0, "market": "pra", "line": 25}).json()
    assert combo["method"].endswith("copula simulations")


def test_props_validation(client):
    assert client.post("/api/props", json={"player_id": 0, "market": "XYZ", "line": 1}).status_code == 422
    assert client.post("/api/props", json={"player_id": 0, "line": 1, "over_odds": 50}).status_code == 422


def test_compare(client):
    body = client.post(
        "/api/compare", json={"players": [{"player_id": 0}, {"player_id": 100}], "scoring": "yahoo"}
    ).json()
    p = body["p_beats"]
    assert p[0][1] + p[1][0] == pytest.approx(1, abs=1e-6)
    assert sum(x["p_best"] for x in body["players"]) == pytest.approx(1)


def test_slate(client):
    body = client.get("/api/slate", params={"min_minutes": 0}).json()
    assert len(body["games"]) == 2
    assert body["players"]
    fantasy = [p["fantasy"]["espn"] for p in body["players"]]
    assert fantasy == sorted(fantasy, reverse=True)


def test_unknown_player_is_404(client):
    assert client.post("/api/projections", json={"player_id": 424242}).status_code == 404
