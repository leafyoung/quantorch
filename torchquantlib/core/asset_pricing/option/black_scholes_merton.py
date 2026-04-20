import math

import torch
from torch import Tensor


def _deterministic_option_value(
    option_type: str,
    spot: Tensor,
    strike: Tensor,
    expiry: Tensor,
    rate: Tensor,
    dividend: Tensor,
) -> Tensor:
    forward_spot = spot * torch.exp((rate - dividend) * expiry)
    discount_factor = torch.exp(-rate * expiry)

    if option_type == "call":
        return discount_factor * torch.maximum(forward_spot - strike, torch.zeros_like(spot))
    return discount_factor * torch.maximum(strike - forward_spot, torch.zeros_like(spot))


def black_scholes_merton(
    option_type: str,
    option_style: str,
    spot: Tensor,
    strike: Tensor,
    expiry: Tensor,
    volatility: Tensor,
    rate: Tensor,
    dividend: Tensor,
) -> Tensor:
    """Price a Black-Scholes-Merton option.

    ``option_style='american'`` uses a simple lower-bounded approximation: the
    returned value is at least the European price and at least the intrinsic value.
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")
    if option_style not in {"european", "american"}:
        raise ValueError("option_style must be either 'european' or 'american'.")
    if volatility.item() <= 0:
        european_price = _deterministic_option_value(option_type, spot, strike, expiry, rate, dividend)
    else:
        sqrt_expiry = torch.sqrt(expiry)
        d1 = (torch.log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * expiry) / (volatility * sqrt_expiry)
        d2 = d1 - volatility * sqrt_expiry

        normal_cdf_d1 = 0.5 * (1 + torch.erf(d1 / math.sqrt(2)))
        normal_cdf_d2 = 0.5 * (1 + torch.erf(d2 / math.sqrt(2)))

        if option_type == "call":
            european_price = spot * torch.exp(-dividend * expiry) * normal_cdf_d1 - strike * torch.exp(-rate * expiry) * normal_cdf_d2
        else:
            european_price = strike * torch.exp(-rate * expiry) * (1 - normal_cdf_d2) - spot * torch.exp(-dividend * expiry) * (1 - normal_cdf_d1)

    if option_style == "european":
        return european_price

    intrinsic_value = torch.maximum(spot - strike, torch.zeros_like(spot)) if option_type == "call" else torch.maximum(strike - spot, torch.zeros_like(spot))
    return torch.maximum(european_price, intrinsic_value)


__all__ = ["black_scholes_merton"]
