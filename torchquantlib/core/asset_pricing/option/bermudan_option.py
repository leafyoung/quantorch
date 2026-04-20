"""Bermudan option pricing via a binomial tree."""

from torch import Tensor

from torchquantlib.core.asset_pricing.option.utils import price_binomial_tree


def bermudan_option(
    option_type: str,
    spot: Tensor,
    strike: Tensor,
    expiry: Tensor,
    volatility: Tensor,
    rate: Tensor,
    steps: int,
    exercise_dates: Tensor,
) -> Tensor:
    """Price a Bermudan option with specified exercise dates."""
    if volatility.item() <= 0:
        raise RuntimeError("volatility must be positive for bermudan_option.")
    return price_binomial_tree(
        option_type,
        "bermudan",
        spot,
        strike,
        expiry,
        volatility,
        rate,
        steps,
        exercise_dates=exercise_dates,
    )
