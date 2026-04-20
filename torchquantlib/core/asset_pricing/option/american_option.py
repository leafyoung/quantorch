from torch import Tensor

from torchquantlib.core.asset_pricing.option.utils import price_binomial_tree


def american_option(
    option_type: str,
    spot: Tensor,
    strike: Tensor,
    expiry: Tensor,
    volatility: Tensor,
    rate: Tensor,
    steps: int,
) -> Tensor:
    """Price an American option with a binomial tree."""
    if volatility.item() <= 0:
        raise RuntimeError("volatility must be positive for american_option.")
    return price_binomial_tree(option_type, "american", spot, strike, expiry, volatility, rate, steps)


__all__ = ["american_option"]
