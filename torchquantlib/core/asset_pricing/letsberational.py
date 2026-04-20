import math

import torch
from torch import Tensor


def _ensure_tensor(value, *, device=None, dtype=None) -> Tensor:
    if isinstance(value, Tensor):
        tensor = value
        if device is not None:
            tensor = tensor.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor
    return torch.as_tensor(value, device=device, dtype=dtype or torch.float32)


def normal_cdf(x):
    """Return the standard normal cumulative distribution function."""
    tensor_x = _ensure_tensor(x)
    return 0.5 * (1.0 + torch.erf(tensor_x / math.sqrt(2.0)))


def normal_pdf(x):
    """Return the standard normal probability density function."""
    tensor_x = _ensure_tensor(x)
    coefficient = torch.as_tensor(0.3989, device=tensor_x.device, dtype=tensor_x.dtype)
    return coefficient * torch.exp(-0.5 * tensor_x**2)


def _black_scholes_price(
    volatility: Tensor,
    spot: Tensor,
    strike: Tensor,
    expiry: Tensor,
    rate: Tensor,
    *,
    is_call: bool,
) -> Tensor:
    sqrt_expiry = torch.sqrt(expiry)
    d1 = (torch.log(spot / strike) + (rate + 0.5 * volatility**2) * expiry) / (volatility * sqrt_expiry)
    d2 = d1 - volatility * sqrt_expiry
    discount_factor = torch.exp(-rate * expiry)

    if is_call:
        return spot * normal_cdf(d1) - strike * discount_factor * normal_cdf(d2)
    return strike * discount_factor * normal_cdf(-d2) - spot * normal_cdf(-d1)


def implied_volatility(option_price, spot, strike, expiry, rate, is_call=True):
    """Estimate implied volatility with a robust bisection search."""
    option_price = _ensure_tensor(option_price, dtype=torch.float32)
    spot = _ensure_tensor(spot, device=option_price.device, dtype=option_price.dtype)
    strike = _ensure_tensor(strike, device=option_price.device, dtype=option_price.dtype)
    expiry = _ensure_tensor(expiry, device=option_price.device, dtype=option_price.dtype)
    rate = _ensure_tensor(rate, device=option_price.device, dtype=option_price.dtype)

    if option_price.numel() != 1 or spot.numel() != 1 or strike.numel() != 1 or expiry.numel() != 1 or rate.numel() != 1:
        raise RuntimeError("implied_volatility currently expects scalar tensor inputs.")
    if option_price.item() <= 0:
        raise RuntimeError("option_price must be positive.")
    if spot.item() <= 0 or strike.item() <= 0 or expiry.item() <= 0:
        raise RuntimeError("spot, strike, and expiry must be positive.")

    discount_factor = torch.exp(-rate * expiry)
    max_price = spot if is_call else strike * discount_factor
    if option_price.item() >= max_price.item():
        raise RuntimeError("option_price is outside no-arbitrage bounds.")

    lower = torch.as_tensor(1e-5, device=option_price.device, dtype=option_price.dtype)
    upper = torch.as_tensor(5.0, device=option_price.device, dtype=option_price.dtype)
    tolerance = 1e-8

    lower_price = _black_scholes_price(lower, spot, strike, expiry, rate, is_call=True)
    if option_price.item() <= lower_price.item():
        return lower.detach()

    for _ in range(100):
        mid = 0.5 * (lower + upper)
        model_price = _black_scholes_price(mid, spot, strike, expiry, rate, is_call=True)
        price_diff = model_price - option_price

        if torch.abs(price_diff).item() < tolerance:
            return mid.detach()

        if price_diff.item() > 0:
            upper = mid
        else:
            lower = mid

    raise RuntimeError("implied_volatility failed to converge.")
