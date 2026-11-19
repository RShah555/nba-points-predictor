import pytest

from nba_predictor.quant import (
    american_to_decimal,
    expected_value,
    implied_probability,
    kelly_fraction,
    price_prop,
    remove_vig,
)


def test_odds_conversion():
    assert american_to_decimal(-110) == pytest.approx(1.9090909)
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert implied_probability(-200) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        american_to_decimal(50)


def test_vig_removal_is_proportional():
    over, under = remove_vig(implied_probability(-110), implied_probability(-110))
    assert over == pytest.approx(0.5) and under == pytest.approx(0.5)
    over, under = remove_vig(0.6, 0.45)
    assert over + under == pytest.approx(1.0)
    assert over / under == pytest.approx(0.6 / 0.45)


def test_expected_value_and_kelly_match_textbook_values():
    # 55% at even money: EV = 0.1 per unit, Kelly = 2p - 1 = 10%
    assert expected_value(0.55, 100) == pytest.approx(0.10)
    assert kelly_fraction(0.55, 100) == pytest.approx(0.10)
    # no edge, no bet
    assert kelly_fraction(0.5, -110) == 0.0


def test_kelly_with_pushes_only_counts_decided_bets():
    # 50% win, 10% push, 40% lose at even money: f = (p - q) / (p + q) = 0.1 / 0.9
    assert kelly_fraction(0.5, 100, p_push=0.1) == pytest.approx(0.1 / 0.9)


def test_price_prop_recommends_the_side_with_an_edge():
    quote = price_prop(0.60, 0.40, 0.0, -110, -110, bankroll=1000, kelly_multiplier=0.25, max_fraction=0.05)
    assert quote.recommendation == "over"
    assert quote.over.edge == pytest.approx(0.10)
    assert quote.over.expected_value > 0 > quote.under.expected_value
    assert 0 < quote.over.kelly_stake <= 50
    assert quote.hold == pytest.approx(2 * implied_probability(-110) - 1)


def test_price_prop_passes_without_enough_edge():
    assert price_prop(0.51, 0.49, 0.0, -110, -110).recommendation == "pass"
