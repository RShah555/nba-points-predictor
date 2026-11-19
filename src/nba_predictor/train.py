"""End to end training: data, walk forward evaluation, calibration, backtests and artifacts.

Walk forward folds over the configured seasons (default 2023-24, 2024-25, 2025-26):

    fold 1  train 2023-24            -> validate 2024-25   (hyperparameters are tuned here)
    fold 2  train 2023-24 + 2024-25  -> test 2025-26       (held out: the headline numbers)
    serve   train all three seasons  -> projections for the season in progress

Each fold's dispersion (spread) is fit on the previous fold's out of sample predictions,
so the held out season never informs its own intervals or probabilities.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import __version__
from . import distributions as dist
from .backtest import calibration, point_metrics, prop_backtest
from .config import Settings, get_settings
from .data import load_season
from .fantasy import PRESETS
from .features import FEATURES, TARGETS, build_training_frame
from .modeling import ModelBundle, TargetModel, fit_booster, grouped_importance, tune
from .seasons import season_id

log = logging.getLogger(__name__)

V1_FEATURES = [
    f"{s}_roll_{w}" for s in ("PTS", "MIN", "FGA", "FG3A", "FTA", "AST", "REB", "TOV") for w in (3, 5, 10)
] + ["SEASON_AVG_PTS", "SEASON_AVG_MIN", "GAMES_PLAYED", "HOME", "REST_DAYS", "BACK_TO_BACK", "OPP_PTS_ALLOWED"]
BACKTEST_STATS = ("PTS", "REB", "AST", "FG3M")


@dataclass(frozen=True)
class Fold:
    train: list[str]
    test: str
    tuning: bool


def make_folds(train_seasons: list[str], test_season: str) -> list[Fold]:
    seasons = [*train_seasons, test_season]
    return [Fold(seasons[:i], seasons[i], tuning=(i == len(seasons) - 2)) for i in range(1, len(seasons))]


def _rows(frame: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    return frame[frame["SEASON_ID"].isin([season_id(s) for s in seasons])]


def _scaled_rounds(best_iteration: int, tuned_rows: int, rows: int) -> int:
    """More data supports a few more trees; grow rounds with the square root of the data size."""
    return max(1, int(best_iteration * float(np.clip(np.sqrt(rows / tuned_rows), 1.0, 1.5))))


def run(settings: Settings | None = None, trials: int | None = None, use_mlflow: bool = True) -> dict[str, Any]:
    settings = settings or get_settings()
    trials = settings.optuna_trials if trials is None else trials
    started = time.time()
    rng = np.random.default_rng(settings.random_state)

    seasons = [*settings.train_seasons, settings.test_season]
    raw = pd.concat([load_season(s, settings=settings) for s in seasons], ignore_index=True)
    frame, _ = build_training_frame(raw)
    folds = make_folds(settings.train_seasons, settings.test_season)
    tuning_fold = next(f for f in folds if f.tuning)
    test_fold = folds[-1]
    test = _rows(frame, [test_fold.test])
    test_train = _rows(frame, test_fold.train)
    log.info("%s player games: training on %s, testing on %s", len(frame), test_fold.train, test_fold.test)

    models: dict[str, TargetModel] = {}
    fold_metrics: list[dict[str, Any]] = [
        {
            "train": f.train,
            "test": f.test,
            "role": "validation (tuning)" if f.tuning else "held out test",
            "rows": len(_rows(frame, [f.test])),
            "targets": {},
        }
        for f in folds
    ]
    test_mu: dict[str, np.ndarray] = {}
    test_models: dict[str, dist.StatModel] = {}
    book_models: dict[str, dist.StatModel] = {}
    serve_k: dict[str, np.ndarray] = {}
    tuning_log: dict[str, Any] = {}

    for target in TARGETS:
        t0 = time.time()
        tr, va = _rows(frame, tuning_fold.train), _rows(frame, [tuning_fold.test])
        res = tune(tr, tr[target], va, va[target], trials, settings.random_state)
        tuning_log[target] = {
            "params": res.params,
            "best_iteration": res.best_iteration,
            "val_poisson_deviance": round(res.val_poisson_deviance, 5),
            "trials": res.trials,
        }

        previous_oos: tuple[np.ndarray, np.ndarray] | None = None
        previous_base: tuple[np.ndarray, np.ndarray] | None = None
        for i, fold in enumerate(folds):
            ftr, fte = _rows(frame, fold.train), _rows(frame, [fold.test])
            rounds = _scaled_rounds(res.best_iteration, len(tr), len(ftr))
            booster = fit_booster(res.params, ftr, ftr[target], rounds)
            mu = np.asarray(booster.predict(fte[FEATURES].astype(float)))
            y = fte[target].to_numpy(dtype=float)
            # spread and recalibration come from the previous fold (first fold: fit on itself)
            stat_model = dist.fit_stat_model(*(previous_oos or (y, mu)), rng)
            k = stat_model.dispersion.size(mu)
            fold_metrics[i]["targets"][target] = {
                **point_metrics(y, mu),
                "season_avg": point_metrics(y, fte[f"SEASON_AVG_{target}"].to_numpy()),
                "last_5": point_metrics(y, fte[f"{target}_roll_5"].to_numpy()),
                **calibration(y, mu, k, rng, stat_model.recalibration),
            }
            base = fte[f"SEASON_AVG_{target}"].to_numpy(dtype=float)
            if fold is test_fold:
                test_mu[target], test_models[target] = mu, stat_model
                # the synthetic book prices from the season average, calibrated on its own past errors
                book_models[target] = dist.fit_stat_model(*(previous_base or (y, base)), rng)
            previous_oos = (y, mu)
            previous_base = (y, base)

        # Serving model: every season, distribution fit on the held out fold's predictions.
        y_last, mu_last = previous_oos  # type: ignore[misc]
        serve_model = dist.fit_stat_model(y_last, mu_last, rng)
        serve_k[target] = serve_model.dispersion.size(mu_last)
        everything = _rows(frame, seasons)
        booster = fit_booster(
            res.params, everything, everything[target], _scaled_rounds(res.best_iteration, len(tr), len(everything))
        )
        models[target] = TargetModel.from_booster(target, booster, serve_model, res.params)
        log.info("%-5s MAE %.3f  (%.0fs)", target, fold_metrics[-1]["targets"][target]["MAE"], time.time() - t0)

    # Copula: how the stats move together, from held out residuals.
    pits = np.column_stack(
        [
            dist.randomized_pit(test[t].to_numpy(dtype=float), test_mu[t], serve_k[t], rng, models[t].recalibration)
            for t in TARGETS
        ]
    )
    copula = dist.fit_copula(pits)

    # The v1 model on the same split, as a reference point.
    v1 = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    ).fit(test_train[V1_FEATURES], test_train["PTS"])
    y_pts = test["PTS"].to_numpy(dtype=float)
    headline = {
        "Season average so far": point_metrics(y_pts, test["SEASON_AVG_PTS"].to_numpy()),
        "Last 5 game average": point_metrics(y_pts, test["PTS_roll_5"].to_numpy()),
        "v1 gradient boosting": point_metrics(y_pts, v1.predict(test[V1_FEATURES])),
        "v2 LightGBM + Optuna": point_metrics(y_pts, test_mu["PTS"]),
    }

    fantasy = {}
    for key in ("espn", "yahoo"):  # linear scoring, so expected fantasy points = weights . means
        w = PRESETS[key].weights
        actual = np.sum([wt * test[s].to_numpy(dtype=float) for s, wt in w.items()], axis=0)
        model = np.sum([wt * test_mu[s] for s, wt in w.items()], axis=0)
        base = np.sum([wt * test[f"SEASON_AVG_{s}"].to_numpy(dtype=float) for s, wt in w.items()], axis=0)
        fantasy[PRESETS[key].name] = {"model": point_metrics(actual, model), "season_avg": point_metrics(actual, base)}

    backtests = {s: prop_backtest(test, s, test_mu[s], test_models[s], book=book_models[s]) for s in BACKTEST_STATS}

    best_base = min(headline["Season average so far"]["MAE"], headline["Last 5 game average"]["MAE"])
    report: dict[str, Any] = {
        "version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "train_seasons": test_fold.train,
        "test_season": test_fold.test,
        "serving_seasons": seasons,
        "train_rows": len(test_train),
        "test_rows": len(test),
        "n_features": len(FEATURES),
        "optuna_trials": trials,
        "headline": headline,
        "improvement_pct": round(100 * (best_base - headline["v2 LightGBM + Optuna"]["MAE"]) / best_base, 1),
        "improvement_vs_v1_pct": round(
            100
            * (headline["v1 gradient boosting"]["MAE"] - headline["v2 LightGBM + Optuna"]["MAE"])
            / headline["v1 gradient boosting"]["MAE"],
            1,
        ),
        "targets": fold_metrics[-1]["targets"],
        "folds": fold_metrics,
        "fantasy": fantasy,
        "backtests": backtests,
        "importance": grouped_importance(models["PTS"]),
        "dispersion": {
            t: {"c0": round(m.dispersion.c0, 4), "c1": round(m.dispersion.c1, 4)} for t, m in models.items()
        },
        "copula": {"stats": list(TARGETS), "correlation": copula.round(4).tolist()},
        "tuning": tuning_log,
        "duration_seconds": round(time.time() - started, 1),
    }

    bundle = ModelBundle(
        __version__,
        report["trained_at"],
        models,
        copula,
        seasons,
        {k: report[k] for k in ("headline", "targets", "test_season", "train_seasons", "improvement_pct", "fantasy")},
    )
    write_artifacts(bundle, report, settings)
    if use_mlflow:
        log_to_mlflow(report, settings)
    return report


def write_artifacts(bundle: ModelBundle, report: dict[str, Any], settings: Settings) -> None:
    out = settings.artifacts_dir
    out.mkdir(parents=True, exist_ok=True)
    bundle.save(settings.bundle_path)
    settings.report_path.write_text(json.dumps(report, indent=2))
    summary = {
        k: report[k]
        for k in (
            "version",
            "trained_at",
            "train_seasons",
            "test_season",
            "train_rows",
            "test_rows",
            "headline",
            "improvement_pct",
            "improvement_vs_v1_pct",
        )
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    (out / "model_card.md").write_text(model_card(report))
    readme = settings.artifacts_dir.parent / "README.md"
    if readme.exists():
        update_readme(readme, report)


def log_to_mlflow(report: dict[str, Any], settings: Settings) -> None:
    try:
        import mlflow
    except ImportError:
        log.info("mlflow is not installed; skipping experiment tracking (pip install -e .[mlops])")
        return
    Path(settings.mlflow_tracking_uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("nba-predictor")
    with mlflow.start_run(run_name=f"v{report['version']} {report['trained_at']}"):
        mlflow.log_params(
            {
                "test_season": report["test_season"],
                "train_seasons": ",".join(report["train_seasons"]),
                "optuna_trials": report["optuna_trials"],
                "n_features": report["n_features"],
            }
        )
        for target, t in report["tuning"].items():
            mlflow.log_params({f"{target}.{k}": v for k, v in t["params"].items()})
        for target, m in report["targets"].items():
            mlflow.log_metrics(
                {
                    f"{target}_MAE": m["MAE"],
                    f"{target}_RMSE": m["RMSE"],
                    f"{target}_coverage_80": m["coverage_80"],
                    f"{target}_log_likelihood": m["log_likelihood"],
                }
            )
        mlflow.log_metric("PTS_improvement_pct", report["improvement_pct"])
        mlflow.log_artifact(str(settings.report_path))
        mlflow.log_artifact(str(settings.artifacts_dir / "model_card.md"))


def results_table(report: dict[str, Any]) -> str:
    lines = [f"{'Method':<26}{'MAE':>7}{'RMSE':>8}"]
    for name, m in report["headline"].items():
        lines.append(f"{name:<26}{m['MAE']:>7.2f}{m['RMSE']:>8.2f}")
    return "\n".join(lines)


def stat_table(report: dict[str, Any]) -> str:
    lines = [f"{'Stat':<6}{'Model MAE':>10}{'Season avg':>12}{'Last 5':>9}{'80% cover':>11}"]
    for t, m in report["targets"].items():
        lines.append(
            f"{t:<6}{m['MAE']:>10.3f}{m['season_avg']['MAE']:>12.3f}{m['last_5']['MAE']:>9.3f}{m['coverage_80']:>11.1%}"
        )
    return "\n".join(lines)


def _markdown_results(report: dict[str, Any]) -> str:
    pts = report["targets"]["PTS"]
    bt = report["backtests"]["PTS"]
    lines = [
        f"Held out test: {report['test_rows']:,} player games from {report['test_season']}, "
        f"trained on {report['train_rows']:,} games from {' and '.join(report['train_seasons'])}.",
        "",
        "**Points**",
        "",
        "| Method | MAE (points) | RMSE (points) |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {n} | {m['MAE']:.2f} | {m['RMSE']:.2f} |" for n, m in report["headline"].items()]
    lines += [
        "",
        f"The v2 model's average miss is {report['improvement_pct']}% smaller than the best baseline "
        f"and {report['improvement_vs_v1_pct']}% smaller than v1.",
        "",
        "**Every stat** (MAE, and how often the actual result landed inside the 80% interval)",
        "",
        "| Stat | Model | Season avg | Last 5 | 80% interval coverage |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {t} | {m['MAE']:.2f} | {m['season_avg']['MAE']:.2f} | {m['last_5']['MAE']:.2f} | {m['coverage_80']:.1%} |"
        for t, m in report["targets"].items()
    ]
    lines += [
        "",
        "**Fantasy points** (expected fantasy points vs actual)",
        "",
        "| Scoring | Model MAE | Season avg MAE |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {n} | {f['model']['MAE']:.2f} | {f['season_avg']['MAE']:.2f} |" for n, f in report["fantasy"].items()]
    lines += [
        "",
        f"**Calibration (points).** Log likelihood per game {pts['log_likelihood']} (recalibrated), "
        f"{pts['nb_log_likelihood']} (raw negative binomial), {pts['poisson_log_likelihood']} (plain Poisson). "
        f"Largest PIT bin deviation from a perfect 10%: {pts['pit_max_deviation']:.1%} "
        f"(raw: {max(abs(b - 0.1) for b in pts['pit_histogram_raw']):.1%}). "
        f"Over/under Brier score {bt['brier_model']} vs 0.25 for a coin flip.",
        "",
        f"**Prop backtest (points, synthetic lines).** {bt['games']:,} games of players averaging 20+ minutes, "
        "priced by a baseline book that knows only the season average (calibrated on its own past errors, "
        f"4.5% margin). The model bet {bt['bets']:,} of them, quarter Kelly, at most 2% per bet and 20% per day, "
        f"no compounding: hit rate {bt['hit_rate']:.1%}, ROI on stakes {bt['roi']:.1%}, "
        f"max drawdown {bt['max_drawdown']:.1%}. This shows the model's probabilities beat a season average "
        "line, not that it beats sportsbooks, which also price in injuries, lineups and market money.",
    ]
    return "\n".join(lines)


def update_readme(path: Path, report: dict[str, Any]) -> None:
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        return
    before, rest = text.split(start, 1)
    after = rest.split(end, 1)[1]
    path.write_text(before + start + "\n" + _markdown_results(report) + "\n" + end + after, encoding="utf-8")


def model_card(report: dict[str, Any]) -> str:
    return f"""# Model card: NBA player projections v{report["version"]}

Trained {report["trained_at"]}.

## What it does
Predicts the full probability distribution of a player's {", ".join(report["targets"])} in his next game.
Each stat has a LightGBM model (Poisson objective) for its mean and a negative binomial distribution whose
spread depends on the mean. A Gaussian copula ties the stats together for fantasy points and combo props.

## Data
NBA.com regular season player game logs via `nba_api` (one LeagueGameLog call per season).
Held out evaluation: trained on {", ".join(report["train_seasons"])} ({report["train_rows"]:,} games),
tested on {report["test_season"]} ({report["test_rows"]:,} games). The serving model is refit on
{", ".join(report["serving_seasons"])}. {report["n_features"]} features, all computed only from games before
the one being predicted.

## Results on {report["test_season"]}
```
{results_table(report)}

{stat_table(report)}
```

## Intended use
Fantasy start/sit decisions, projections and learning. The prop pricing tools are informational.
The betting backtest uses synthetic lines (season average so far), which are much easier to beat than
real sportsbook lines.

## Limitations
* No injury, lineup or minutes news: if a teammate is ruled out an hour before tip, the model does not know.
* Players with fewer than 5 games in a season are excluded from training and evaluation.
* Stat models are trained separately, so simulated PTS is not forced to equal 2*FGM + FG3M + FTM.
* Opponent strength is measured by what a defense allowed, not by matchup or position.
"""


def main(trials: int | None = None, use_mlflow: bool = True) -> None:
    report = run(trials=trials, use_mlflow=use_mlflow)
    print(
        f"\nTraining on {report['train_rows']:,} player games, testing on {report['test_rows']:,} "
        f"from {report['test_season']}\n"
    )
    print(results_table(report))
    print(
        f"\nModel beats the best baseline by {report['improvement_pct']}% on MAE "
        f"and v1 by {report['improvement_vs_v1_pct']}%\n"
    )
    print(stat_table(report))
    bt = report["backtests"]["PTS"]
    print(
        f"\nPoints prop backtest (synthetic lines): {bt['bets']:,} bets, hit rate {bt['hit_rate']:.1%}, "
        f"ROI {bt['roi']:.1%}, Brier {bt['brier_model']} vs 0.25 coin flip"
    )
    print(
        f"\nSaved models/nba_predictor.joblib, models/report.json, models/metrics.json, models/model_card.md "
        f"in {report['duration_seconds']:.0f}s"
    )
