"""Evaluation: point accuracy, calibration and a simulated prop betting backtest."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import distributions as dist


def point_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": round(float(mean_absolute_error(y, pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, pred))), 3),
    }


def calibration(
    y: np.ndarray, mu: np.ndarray, k: np.ndarray, rng: np.random.Generator, recal: dist.Recalibration = dist.IDENTITY
) -> dict[str, Any]:
    """Interval coverage, PIT histograms (before and after recalibration) and log likelihoods.

    Stats are whole numbers, so an interval with inclusive ends covers at least its nominal
    level; low count stats (steals, blocks) cover noticeably more. The PIT histogram is the
    sharper check: ten bins of 10% each when the distribution is calibrated.
    """
    lo80, hi80 = dist.ppf(0.1, mu, k, recal), dist.ppf(0.9, mu, k, recal)
    lo50, hi50 = dist.ppf(0.25, mu, k, recal), dist.ppf(0.75, mu, k, recal)

    def hist(pit: np.ndarray) -> list[float]:
        h, _ = np.histogram(pit, bins=10, range=(0, 1))
        return (h / h.sum()).round(4).tolist()

    raw_pit = dist.randomized_pit(y, mu, k, rng)
    pit = dist.randomized_pit(y, mu, k, rng, recal)
    probs = np.clip(dist.cdf(y, mu, k, recal) - dist.cdf(y - 1, mu, k, recal), 1e-12, None)
    poisson_ll = dist.nb_logpmf(y, mu, np.full_like(mu, 1e6))
    return {
        "coverage_80": round(float(np.mean((y >= lo80) & (y <= hi80))), 4),
        "coverage_50": round(float(np.mean((y >= lo50) & (y <= hi50))), 4),
        "pit_histogram": hist(pit),
        "pit_histogram_raw": hist(raw_pit),
        "pit_max_deviation": round(float(np.max(np.abs(np.array(hist(pit)) - 0.1))), 4),
        "log_likelihood": round(float(np.mean(np.log(probs))), 4),
        "nb_log_likelihood": round(float(np.mean(dist.nb_logpmf(y, mu, k))), 4),
        "poisson_log_likelihood": round(float(np.mean(poisson_ll)), 4),
    }


def _max_drawdown(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity)
    return float(np.max((peaks - equity) / peaks)) if len(equity) else 0.0


def baseline_line(base_mu: np.ndarray, model: dist.StatModel) -> np.ndarray:
    """The half point line a simple book would hang: its median, using the season average as the mean.

    Using the median (not the mean) matters: points are right skewed, so a line at the
    season average would make the under a winner most nights for any model at all.
    """
    k = model.dispersion.size(base_mu)
    median = dist.ppf(0.5, base_mu, k, model.recalibration)
    lower, upper = median - 0.5, median + 0.5
    p_lower = 1 - dist.cdf(median - 1, base_mu, k, model.recalibration)
    p_upper = 1 - dist.cdf(median, base_mu, k, model.recalibration)
    line = np.where(np.abs(p_lower - 0.5) <= np.abs(p_upper - 0.5), lower, upper)
    return np.maximum(line, 0.5)


def prop_backtest(
    frame: pd.DataFrame,
    stat: str,
    mu: np.ndarray,
    model: dist.StatModel,
    *,
    overround: float = 1.0455,
    bankroll: float = 1000.0,
    kelly_multiplier: float = 0.25,
    max_fraction: float = 0.02,
    max_daily_exposure: float = 0.20,
    min_edge: float = 0.03,
    min_minutes: float = 20.0,
    book: dist.StatModel | None = None,
) -> dict[str, Any]:
    """Bet against a synthetic baseline book and track a bankroll day by day.

    Real historical sportsbook lines are not freely available, so each game is priced by a
    baseline book that knows the player's season average and the shape of his distribution:
    the line is the half point nearest its median, and each side is priced at the book's fair
    probability times an overround of 1.0455 (the margin in -110/-110). A model with no
    information beyond the season average therefore finds no positive EV bets. Real books
    also use news and market information, so treat the ROI as an upper bound, not a forecast.
    The calibration numbers (Brier score, reliability) are the more honest signal.

    Only players averaging `min_minutes` (the ones books post props for) are bet, each bet
    is capped at `max_fraction` of the bankroll and each day's total at `max_daily_exposure`.
    Stakes are sized off the starting bankroll (no compounding), so the curve is cumulative
    profit rather than an exponential that exaggerates a small edge over thousands of bets.
    """
    keep = (frame["SEASON_AVG_MIN"] >= min_minutes).to_numpy()
    frame, mu = frame[keep], mu[keep]
    base_mu = frame[f"SEASON_AVG_{stat}"].to_numpy(dtype=float)
    # The book's spread and calibration are fit on the season average's own past errors when given,
    # so the book is honest about its own uncertainty rather than borrowing the model's.
    book = book or model
    line = baseline_line(base_mu, book)
    book_over = 1 - dist.cdf(np.floor(line), base_mu, book.dispersion.size(base_mu), book.recalibration)
    actual = frame[stat].to_numpy()
    k = model.dispersion.size(mu)
    p_over = 1 - dist.cdf(np.floor(line), mu, k, model.recalibration)
    went_over = actual > line

    brier = float(np.mean((p_over - went_over) ** 2))
    bins = np.clip((p_over * 10).astype(int), 0, 9)
    reliability = [
        {
            "bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}",
            "predicted": round(float(p_over[bins == b].mean()), 4),
            "observed": round(float(went_over[bins == b].mean()), 4),
            "n": int((bins == b).sum()),
        }
        for b in range(10)
        if (bins == b).sum() >= 50
    ]

    bet_over = p_over >= book_over
    p_side = np.where(bet_over, p_over, 1 - p_over)
    fair = np.where(bet_over, book_over, 1 - book_over)
    b = 1 / np.clip(fair * overround, 1e-6, 0.999) - 1  # decimal odds minus one, per game
    edge = p_side - fair
    ev = p_side * b - (1 - p_side)
    kelly = np.clip((b * p_side - (1 - p_side)) / b, 0, None) * kelly_multiplier
    fraction = np.minimum(kelly, max_fraction)
    take = (ev > 0) & (edge >= min_edge)
    won = np.where(bet_over, went_over, ~went_over)

    bets = pd.DataFrame(
        {"date": frame["GAME_DATE"].to_numpy(), "fraction": fraction, "won": won, "edge": edge, "b": b}
    )[take]
    equity, staked, profit, curve = bankroll, 0.0, 0.0, []
    for day, games in bets.groupby("date", sort=True):
        fractions = games["fraction"].to_numpy()
        fractions = fractions * min(1.0, max_daily_exposure / fractions.sum())
        stakes = fractions * bankroll  # sized off the starting bankroll: no compounding
        pnl = float(np.sum(np.where(games["won"], stakes * games["b"].to_numpy(), -stakes)))
        staked += float(stakes.sum())
        profit += pnl
        equity += pnl
        curve.append({"date": pd.Timestamp(day).strftime("%Y-%m-%d"), "bankroll": round(equity, 2)})

    buckets = []
    for lo, hi in ((0.02, 0.04), (0.04, 0.07), (0.07, 0.10), (0.10, 1.0)):
        sel = bets[(bets["edge"] >= lo) & (bets["edge"] < hi)]
        if len(sel):
            flat = np.where(sel["won"], sel["b"], -1.0)
            buckets.append(
                {
                    "edge": f"{lo:.0%}-{hi:.0%}" if hi < 1 else f"{lo:.0%}+",
                    "bets": len(sel),
                    "hit_rate": round(float(sel["won"].mean()), 4),
                    "flat_roi": round(float(flat.mean()), 4),
                }
            )

    equity_series = np.array([bankroll] + [c["bankroll"] for c in curve])
    return {
        "stat": stat,
        "line": "baseline book: season average distribution, median line, fair price + margin",
        "overround": overround,
        "over_rate": round(float(went_over.mean()), 4),
        "kelly_multiplier": kelly_multiplier,
        "max_fraction": max_fraction,
        "min_edge": min_edge,
        "max_daily_exposure": max_daily_exposure,
        "min_minutes": min_minutes,
        "games": len(frame),
        "bets": len(bets),
        "hit_rate": round(float(bets["won"].mean()), 4) if len(bets) else None,
        "roi": round(profit / staked, 4) if staked else None,
        "start_bankroll": bankroll,
        "final_bankroll": round(equity, 2),
        "max_drawdown": round(_max_drawdown(equity_series), 4),
        "brier_model": round(brier, 4),
        "brier_coin_flip": 0.25,
        "reliability": reliability,
        "edge_buckets": buckets,
        "equity_curve": curve,
    }
