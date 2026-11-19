import numpy as np
import pandas as pd

from nba_predictor import distributions as dist
from nba_predictor.backtest import baseline_line, prop_backtest
from nba_predictor.train import make_folds


def test_walk_forward_folds():
    folds = make_folds(["2023-24", "2024-25"], "2025-26")
    assert [(f.train, f.test, f.tuning) for f in folds] == [
        (["2023-24"], "2024-25", True),
        (["2023-24", "2024-25"], "2025-26", False),
    ]


def test_baseline_line_sits_at_the_median():
    model = dist.StatModel(dist.Dispersion(1.0, 0.6))
    mu = np.array([3.0, 12.0, 27.0])
    line = baseline_line(mu, model)
    k = model.dispersion.size(mu)
    p_over = 1 - dist.cdf(np.floor(line), mu, k)
    assert np.all(line % 1 == 0.5)
    assert np.all(np.abs(p_over - 0.5) < 0.12)


def test_a_model_that_knows_nothing_extra_has_no_edge():
    """If the model's mean equals the baseline's, it should find (almost) nothing to bet."""
    rng = np.random.default_rng(0)
    n = 4000
    mu = rng.uniform(8, 25, n)
    model = dist.StatModel(dist.Dispersion(1.5, 0.5))
    k = model.dispersion.size(mu)
    frame = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime("2026-01-01") + pd.to_timedelta(np.arange(n) // 40, unit="D"),
            "SEASON_AVG_PTS": mu,
            "SEASON_AVG_MIN": 30.0,
            "PTS": rng.negative_binomial(k, k / (k + mu)),
        }
    )
    result = prop_backtest(frame, "PTS", mu, model)
    assert result["bets"] == 0
    assert result["final_bankroll"] == result["start_bankroll"]


def test_a_model_with_real_information_finds_edges():
    """Truth differs from the season average; a model that knows the truth should profit."""
    rng = np.random.default_rng(1)
    n = 6000
    season_avg = rng.uniform(8, 25, n)
    truth = season_avg * rng.uniform(0.7, 1.3, n)
    model = dist.StatModel(dist.Dispersion(1.5, 0.5))
    k = model.dispersion.size(truth)
    frame = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime("2026-01-01") + pd.to_timedelta(np.arange(n) // 40, unit="D"),
            "SEASON_AVG_PTS": season_avg,
            "SEASON_AVG_MIN": 30.0,
            "PTS": rng.negative_binomial(k, k / (k + truth)),
        }
    )
    result = prop_backtest(frame, "PTS", truth, model)
    assert result["bets"] > 1000
    assert result["roi"] > 0
    assert result["brier_model"] < 0.25
