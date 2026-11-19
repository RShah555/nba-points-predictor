import numpy as np
import pytest
from scipy import stats

from nba_predictor import distributions as dist


def test_pmf_sums_to_one_and_mean_matches():
    support, pmf = dist.pmf_table(22.0, 15.0, tail=0.999999)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-5)
    assert float((support * pmf).sum()) == pytest.approx(22.0, rel=1e-3)


def test_logpmf_matches_scipy():
    y, mu, k = np.array([0, 3, 17]), np.array([2.0, 5.0, 20.0]), np.array([1.5, 4.0, 12.0])
    expected = stats.nbinom.logpmf(y, k, k / (k + mu))
    np.testing.assert_allclose(dist.nb_logpmf(y, mu, k), expected, rtol=1e-10)


def test_dispersion_fit_recovers_the_truth():
    rng = np.random.default_rng(1)
    mu = rng.uniform(2, 30, 40_000)
    truth = dist.Dispersion(0.4, 0.6)
    k = truth.size(mu)
    y = rng.negative_binomial(k, k / (k + mu))
    fit = dist.fit_dispersion(y, mu)
    assert fit.c0 == pytest.approx(truth.c0, abs=0.15)
    assert fit.c1 == pytest.approx(truth.c1, abs=0.06)


def test_over_under_probabilities_add_up():
    for line in (19.5, 20.0):
        p_over, p_under, p_push = dist.over_under(line, 21.0, 10.0)
        assert p_over + p_under + p_push == pytest.approx(1.0)
        assert (p_push > 0) == float(line).is_integer()


def test_recalibration_fixes_a_miscalibrated_model():
    """Data is narrower than the model believes; recalibration should flatten the PIT histogram."""
    rng = np.random.default_rng(2)
    mu = rng.uniform(5, 25, 30_000)
    y = rng.poisson(mu)  # truth: Poisson
    k = np.full_like(mu, 3.0)  # model: very overdispersed
    raw = dist.randomized_pit(y, mu, k, rng)
    recal = dist.Recalibration.fit(raw)
    fixed = dist.randomized_pit(y, mu, k, rng, recal)

    def worst_bin(p: np.ndarray) -> float:
        h, _ = np.histogram(p, bins=10, range=(0, 1))
        return float(np.max(np.abs(h / h.sum() - 0.1)))

    assert worst_bin(raw) > 0.05
    assert worst_bin(fixed) < 0.01


def test_identity_recalibration_changes_nothing():
    q = np.array([0.1, 0.5, 0.9])
    np.testing.assert_array_equal(dist.ppf(q, 10, 5), dist.ppf(q, 10, 5, dist.Recalibration()))


def test_copula_samples_keep_marginals_and_correlation():
    rng = np.random.default_rng(3)
    corr = np.array([[1.0, 0.7], [0.7, 1.0]])
    mu, k = np.array([20.0, 6.0]), np.array([12.0, 8.0])
    draws = dist.sample_joint(mu, k, corr, 50_000, rng)
    np.testing.assert_allclose(draws.mean(axis=0), mu, rtol=0.02)
    assert np.corrcoef(draws, rowvar=False)[0, 1] == pytest.approx(0.68, abs=0.05)


def test_nearest_correlation_repairs_an_invalid_matrix():
    bad = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])
    fixed = dist.nearest_correlation(bad)
    assert np.all(np.linalg.eigvalsh(fixed) > 0)
    np.testing.assert_allclose(np.diag(fixed), 1.0)
