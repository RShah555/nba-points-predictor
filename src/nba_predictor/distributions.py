"""Turning a predicted mean into a calibrated probability distribution.

1. Shape. Box score stats are counts whose spread grows with the mean faster than a
   Poisson allows, so each stat gets a negative binomial with mean dependent dispersion:

       Y ~ NegBin(mean=mu, size=k(mu)),  k(mu) = exp(c0 + c1 * log(mu)),  Var = mu + mu^2 / k

   c0 and c1 are fit by maximum likelihood on out of sample predictions.

2. Recalibration. Points come in 2s and 3s and are less skewed than a negative binomial,
   so the raw CDF F is passed through a monotone map G learned from out of sample PIT
   values (Kuleshov et al., 2018): the calibrated CDF is G(F(y)). When the parametric
   shape is already right, G is the identity.

3. Dependence. Stats move together (a hot shooting night lifts PTS, FGM and 3PM), so joint
   draws use a Gaussian copula fit on the normal scores of each stat's calibrated PITs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats

MU_FLOOR = 1e-3


@dataclass(frozen=True)
class Dispersion:
    c0: float
    c1: float

    def size(self, mu: np.ndarray | float) -> np.ndarray:
        mu = np.clip(np.asarray(mu, dtype=float), MU_FLOOR, None)
        return np.clip(np.exp(self.c0 + self.c1 * np.log(mu)), 1e-2, 1e6)


@dataclass(frozen=True)
class Recalibration:
    """Piecewise linear monotone map from raw CDF values to calibrated ones."""

    raw: tuple[float, ...] = (0.0, 1.0)
    calibrated: tuple[float, ...] = (0.0, 1.0)

    @classmethod
    def fit(cls, pits: np.ndarray, knots: int = 101) -> Recalibration:
        levels = np.linspace(0, 1, knots)
        raw = np.quantile(np.asarray(pits, dtype=float), levels)
        raw[0], raw[-1] = 0.0, 1.0
        raw = np.maximum.accumulate(raw + np.arange(knots) * 1e-9)  # strictly increasing
        raw = raw / raw[-1]
        return cls(tuple(raw.tolist()), tuple(levels.tolist()))

    def forward(self, u: np.ndarray | float) -> np.ndarray:
        return np.asarray(np.interp(u, self.raw, self.calibrated))

    def inverse(self, q: np.ndarray | float) -> np.ndarray:
        return np.asarray(np.interp(q, self.calibrated, self.raw))


IDENTITY = Recalibration()


@dataclass(frozen=True)
class StatModel:
    """Everything needed to turn a mean into a distribution for one stat."""

    dispersion: Dispersion
    recalibration: Recalibration = field(default_factory=Recalibration)


def _nb_args(mu: np.ndarray | float, k: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    mu = np.clip(np.asarray(mu, dtype=float), MU_FLOOR, None)
    k = np.asarray(k, dtype=float)
    return k, k / (k + mu)


def nb_logpmf(y: np.ndarray, mu: np.ndarray, k: np.ndarray) -> np.ndarray:
    mu = np.clip(mu, MU_FLOOR, None)
    return (
        special.gammaln(y + k)
        - special.gammaln(k)
        - special.gammaln(y + 1)
        + k * np.log(k / (k + mu))
        + y * np.log(mu / (k + mu))
    )


def fit_dispersion(y: np.ndarray, mu: np.ndarray) -> Dispersion:
    """Maximum likelihood fit of the dispersion curve given observed counts and predicted means."""
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), MU_FLOOR, None)
    log_mu = np.log(mu)

    def nll(theta: np.ndarray) -> float:
        k = np.clip(np.exp(theta[0] + theta[1] * log_mu), 1e-2, 1e6)
        return -float(np.mean(nb_logpmf(y, mu, k)))

    res = optimize.minimize(nll, x0=np.array([1.0, 0.5]), method="L-BFGS-B", bounds=[(-6.0, 14.0), (-3.0, 3.0)])
    return Dispersion(float(res.x[0]), float(res.x[1]))


def cdf(
    y: np.ndarray | float, mu: np.ndarray | float, k: np.ndarray | float, recal: Recalibration = IDENTITY
) -> np.ndarray:
    n, p = _nb_args(mu, k)
    raw = np.asarray(stats.nbinom.cdf(y, n, p))
    return raw if recal is IDENTITY else recal.forward(raw)


def ppf(
    q: np.ndarray | float, mu: np.ndarray | float, k: np.ndarray | float, recal: Recalibration = IDENTITY
) -> np.ndarray:
    """Smallest y with G(F(y)) >= q, i.e. F(y) >= G^-1(q)."""
    n, p = _nb_args(mu, k)
    raw_q = q if recal is IDENTITY else recal.inverse(q)
    return np.asarray(stats.nbinom.ppf(np.clip(raw_q, 1e-9, 1 - 1e-9), n, p))


def pmf_table(
    mu: float, k: float, recal: Recalibration = IDENTITY, tail: float = 0.999
) -> tuple[np.ndarray, np.ndarray]:
    """Support 0..ppf(tail) and the probability of each value, for charts."""
    top = int(ppf(tail, mu, k, recal)) + 1
    support = np.arange(0, max(top, 1) + 1)
    c = cdf(support, mu, k, recal)
    return support, np.diff(np.concatenate([[0.0], c]))


def over_under(line: float, mu: float, k: float, recal: Recalibration = IDENTITY) -> tuple[float, float, float]:
    """P(over), P(under), P(push) for a counting stat against a line."""
    if float(line).is_integer():
        at_or_below = float(cdf(line, mu, k, recal))
        p_under = float(cdf(line - 1, mu, k, recal))
        p_push = at_or_below - p_under
    else:
        p_push = 0.0
        p_under = float(cdf(np.floor(line), mu, k, recal))
    return max(0.0, 1.0 - p_under - p_push), p_under, p_push


def randomized_pit(
    y: np.ndarray, mu: np.ndarray, k: np.ndarray, rng: np.random.Generator, recal: Recalibration = IDENTITY
) -> np.ndarray:
    """Probability integral transform for a discrete variable; Uniform(0, 1) if calibrated."""
    hi = cdf(y, mu, k, recal)
    lo = cdf(y - 1, mu, k, recal)
    return np.clip(lo + rng.uniform(size=np.shape(y)) * (hi - lo), 1e-6, 1 - 1e-6)


def fit_stat_model(y: np.ndarray, mu: np.ndarray, rng: np.random.Generator) -> StatModel:
    """Dispersion, then a recalibration map on top of it, from out of sample (y, mu) pairs."""
    disp = fit_dispersion(y, mu)
    pits = randomized_pit(y, mu, disp.size(mu), rng)
    return StatModel(disp, Recalibration.fit(pits))


def nearest_correlation(corr: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues so the matrix is a valid correlation matrix."""
    corr = (corr + corr.T) / 2
    vals, vecs = np.linalg.eigh(corr)
    fixed = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    out = fixed / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return out


def fit_copula(pits: np.ndarray) -> np.ndarray:
    """Correlation of normal scores; `pits` is (games x stats)."""
    z = stats.norm.ppf(pits)
    return nearest_correlation(np.corrcoef(z, rowvar=False))


def sample_joint(
    mu: np.ndarray,
    k: np.ndarray,
    corr: np.ndarray,
    n: int,
    rng: np.random.Generator,
    recals: list[Recalibration] | None = None,
) -> np.ndarray:
    """n joint draws (n x stats) with calibrated marginals tied together by the copula."""
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, len(mu))) @ chol.T
    u = stats.norm.cdf(z)
    out = np.empty_like(u)
    for j in range(len(mu)):  # inverse CDF by table lookup: far faster than nbinom.ppf per draw
        recal = recals[j] if recals else IDENTITY
        top = int(ppf(1 - 1e-6, mu[j], k[j], recal)) + 2
        table = cdf(np.arange(top + 1), mu[j], k[j], recal)
        out[:, j] = np.minimum(np.searchsorted(table, u[:, j], side="left"), top)
    return out
