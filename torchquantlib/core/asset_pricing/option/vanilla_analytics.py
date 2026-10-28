"""Analytic Black-Scholes barrier and binary option prices, ported from
tf-quant-finance ``black_scholes/vanilla_prices.py`` (``barrier_price`` and
``binary_price``).

``barrier_price_analytic`` is the continuous-monitoring single-barrier closed
form (Reiner-Rubinstein / Merton-style, the Clewlow-Strickland / Haug mask
formulation used by tff): 12 terms, each an algebraic factor times a normal
CDF, selected by barrier type and by whether the strike is above the barrier.
``binary_price`` is the cash-or-nothing binary paying one unit of cash, with
the dividend-corrected forward ``F = S * exp(-q*T) / DF`` (the FIXED tff
version; ignoring q mis-prices binaries when q > 0).

Numerics are pure f64 scalar math (``float()`` + ``math`` module, no tensor
ops), mirroring the tff formulas branch-for-branch, including the rebate
terms and the ``sqrt_var > 0`` clamp of the binary price. Arguments accept
Python floats or 0-dim tensors (converted via ``float()``) so the module is
drop-in with the tensor-based pricers of this package.

Values verified against QuantLib (AnalyticBarrierEngine on BarrierOption,
CashOrNothingPayoff + AnalyticEuropeanEngine) and against tff itself in
float64; test literals in ``tests/test_vanilla_analytics.py`` come from those
runs.

Rebate convention (tff semantics): for knock-outs the rebate is paid at hit,
for knock-ins at expiry. QuantLib's AnalyticBarrierEngine agrees with tff on
knock-in rebates but pays knock-out rebates at expiry, so knock-out-rebate
literals are tff-float64-verified rather than QL-verified.
"""

import math


def _ncdf(x: float) -> float:
    return 0.5 * (math.erf(x / math.sqrt(2.0)) + 1.0)


# Mask rows for strike > barrier, indexed by
# (is_barrier_down << 2) + (is_knock_out << 1) + is_call
# (same rows and order as tff's mask_matrix_greater_strike).
_MASK_GREATER_STRIKE = (
    (1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),  # up and in put
    (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0),  # up and in call
    (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0),  # up and out put
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),  # up and out call
    (0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0),  # down and in put
    (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0),  # down and in call
    (1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0),  # down and out put
    (1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),  # down and out call
)

# Mask rows for strike <= barrier (tff's mask_matrix_lower_strike).
_MASK_LOWER_STRIKE = (
    (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0),  # up and in put
    (0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),  # up and in call
    (1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),  # up and out put
    (1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0),  # up and out call
    (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0),  # down and in put
    (1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),  # down and in call
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),  # down and out put
    (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0),  # down and out call
)


def barrier_price_analytic(
    volatility: float,
    strike: float,
    expiry: float,
    spot: float,
    barrier: float,
    rebate: float,
    discount_rate: float,
    dividend_rate: float,
    is_barrier_down: bool,
    is_knock_out: bool,
    is_call: bool,
) -> float:
    """Price a continuously-monitored single-barrier option in closed form
    (Black-Scholes, tff ``barrier_price`` semantics).

    Arguments follow the tff keyword names: ``discount_rate`` is the
    risk-free rate r, ``dividend_rate`` the continuous dividend yield q.
    The barrier must not be breached at inception (down barrier below
    ``spot``, up barrier above ``spot``), matching the formula's domain.
    ``rebate`` is the fixed cash payout if a knock-out barrier is breached
    (paid at hit) or if a knock-in barrier is never breached (paid at
    expiry). Numeric arguments accept floats or 0-dim tensors; the return
    value is a Python float computed in f64.
    """
    volatility = float(volatility)
    strike = float(strike)
    expiry = float(expiry)
    spot = float(spot)
    barrier = float(barrier)
    rebate = float(rebate)
    discount_rate = float(discount_rate)
    dividend_rate = float(dividend_rate)

    index = (
        ((1 if is_barrier_down else 0) << 2)
        + ((1 if is_knock_out else 0) << 1)
        + (1 if is_call else 0)
    )
    masks = _MASK_GREATER_STRIKE[index] if strike > barrier else _MASK_LOWER_STRIKE[index]

    call_or_put = 1.0 if is_call else -1.0
    below_or_above = 1.0 if is_barrier_down else -1.0

    # Parameters for the six integrals (tff names, minus the unicode).
    sqrt_var = volatility * math.sqrt(expiry)
    mu = (discount_rate - dividend_rate) - (volatility**2) / 2.0
    lamda = 1.0 + mu / (volatility**2)
    x = math.log(spot / strike) / sqrt_var + lamda * sqrt_var
    x1 = math.log(spot / barrier) / sqrt_var + lamda * sqrt_var
    y = math.log(barrier**2 / (spot * strike)) / sqrt_var + lamda * sqrt_var
    y1 = math.log(barrier / spot) / sqrt_var + lamda * sqrt_var
    b = (mu**2 + 2.0 * (volatility**2) * discount_rate) / (volatility**2)
    z = math.log(barrier / spot) / sqrt_var + b * sqrt_var
    a = mu / (volatility**2)

    discount_factors = math.exp(-discount_rate * expiry)
    barriers_ratio = barrier / spot
    spots_term = call_or_put * spot * math.exp(-dividend_rate * expiry)
    strikes_term = call_or_put * strike * discount_factors

    terms = (
        spots_term,
        -strikes_term,
        spots_term,
        -strikes_term,
        spots_term * (barriers_ratio ** (2 * lamda)),
        -strikes_term * (barriers_ratio ** (2 * lamda - 2)),
        spots_term * (barriers_ratio ** (2 * lamda)),
        -strikes_term * (barriers_ratio ** (2 * lamda - 2)),
        rebate * discount_factors,
        -rebate * discount_factors * (barriers_ratio ** (2 * lamda - 2)),
        rebate * (barriers_ratio ** (a + b)),
        rebate * (barriers_ratio ** (a - b)),
    )
    cdfs = (
        _ncdf(call_or_put * x),
        _ncdf(call_or_put * (x - sqrt_var)),
        _ncdf(call_or_put * x1),
        _ncdf(call_or_put * (x1 - sqrt_var)),
        _ncdf(below_or_above * y),
        _ncdf(below_or_above * (y - sqrt_var)),
        _ncdf(below_or_above * y1),
        _ncdf(below_or_above * (y1 - sqrt_var)),
        _ncdf(below_or_above * (x1 - sqrt_var)),
        _ncdf(below_or_above * (y1 - sqrt_var)),
        _ncdf(below_or_above * z),
        _ncdf(below_or_above * (z - 2 * b * sqrt_var)),
    )
    return sum(m * t * c for m, t, c in zip(masks, terms, cdfs))


def binary_price(
    volatility: float,
    strike: float,
    expiry: float,
    spot: float,
    discount_rate: float,
    dividend_rate: float,
    is_call: bool,
) -> float:
    """Price a cash-or-nothing binary option paying one unit of cash at
    expiry (Black-Scholes, tff ``binary_price`` semantics with the
    dividend-corrected forward ``F = S * exp(-q*T) / DF``).

    The call pays 1 if the spot ends above the strike, the put 1 if below,
    so the price is the discounted risk-neutral probability of ending in the
    money. Zero volatility collapses to the discounted step payoff (tff's
    ``sqrt_var > 0`` clamp). Numeric arguments accept floats or 0-dim
    tensors; the return value is a Python float computed in f64.
    """
    volatility = float(volatility)
    strike = float(strike)
    expiry = float(expiry)
    spot = float(spot)
    discount_rate = float(discount_rate)
    dividend_rate = float(dividend_rate)

    discount_factors = math.exp(-discount_rate * expiry)
    forwards = spot * math.exp(-dividend_rate * expiry) / discount_factors
    sqrt_var = volatility * math.sqrt(expiry)
    if sqrt_var > 0.0:
        d2 = math.log(forwards / strike) / sqrt_var - sqrt_var / 2.0
        undiscounted_calls = _ncdf(d2)
    else:
        undiscounted_calls = 1.0 if forwards > strike else 0.0
    if is_call:
        return discount_factors * undiscounted_calls
    return discount_factors * (1.0 - undiscounted_calls)


__all__ = ["barrier_price_analytic", "binary_price"]
