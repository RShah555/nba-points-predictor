"""Pricing player props: implied probabilities, vig removal, expected value and Kelly sizing."""

from __future__ import annotations

from dataclasses import asdict, dataclass


def american_to_decimal(odds: float) -> float:
    if -100 < odds < 100:
        raise ValueError("American odds must be <= -100 or >= +100")
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)


def implied_probability(odds: float) -> float:
    return 1 / american_to_decimal(odds)


def remove_vig(p_a: float, p_b: float) -> tuple[float, float]:
    """Multiplicative (proportional) devig: scale both implied probabilities to sum to 1."""
    total = p_a + p_b
    return p_a / total, p_b / total


def expected_value(p_win: float, odds: float, p_push: float = 0.0) -> float:
    """Expected profit per 1 unit staked."""
    p_lose = 1 - p_win - p_push
    return p_win * (american_to_decimal(odds) - 1) - p_lose


def kelly_fraction(p_win: float, odds: float, p_push: float = 0.0) -> float:
    """Growth optimal fraction of bankroll with win/lose/push outcomes; 0 when there is no edge.

    Maximises p log(1 + b f) + q log(1 - f), giving f = (b p - q) / (b (p + q)).
    """
    b = american_to_decimal(odds) - 1
    p, q = p_win, 1 - p_win - p_push
    if p + q <= 0:
        return 0.0
    return max(0.0, (b * p - q) / (b * (p + q)))


@dataclass(frozen=True)
class SideQuote:
    side: str
    odds: float
    model_prob: float
    implied_prob: float
    fair_prob: float
    edge: float
    expected_value: float
    kelly_full: float
    kelly_stake: float


@dataclass(frozen=True)
class PropQuote:
    p_over: float
    p_under: float
    p_push: float
    hold: float
    over: SideQuote
    under: SideQuote
    recommendation: str  # "over", "under" or "pass"

    def to_dict(self) -> dict:
        return asdict(self)


def price_prop(
    p_over: float,
    p_under: float,
    p_push: float,
    over_odds: float,
    under_odds: float,
    bankroll: float = 1000.0,
    kelly_multiplier: float = 0.25,
    max_fraction: float = 0.05,
    min_edge: float = 0.02,
) -> PropQuote:
    """Compare model probabilities with the book's and size a bet with capped fractional Kelly."""
    imp_over, imp_under = implied_probability(over_odds), implied_probability(under_odds)
    fair_over, fair_under = remove_vig(imp_over, imp_under)

    def side(name: str, odds: float, p: float, imp: float, fair: float) -> SideQuote:
        full = kelly_fraction(p, odds, p_push)
        frac = min(full * kelly_multiplier, max_fraction)
        # edge is measured against the no-vig price on the same push-free scale
        edge = p / (1 - p_push) - fair if p_push < 1 else 0.0
        return SideQuote(
            name, odds, p, imp, fair, edge, expected_value(p, odds, p_push), full, round(bankroll * frac, 2)
        )

    over = side("over", over_odds, p_over, imp_over, fair_over)
    under = side("under", under_odds, p_under, imp_under, fair_under)
    best = max((over, under), key=lambda s: s.expected_value)
    rec = best.side if best.expected_value > 0 and best.edge >= min_edge else "pass"
    return PropQuote(p_over, p_under, p_push, imp_over + imp_under - 1, over, under, rec)
