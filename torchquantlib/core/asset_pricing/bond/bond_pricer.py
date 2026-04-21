from __future__ import annotations

import math

import torch
from torch import Tensor


EPSILON = 1e-8


def _period_grid(periods: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
    count = max(1, int(periods.item()))
    return torch.arange(1, count + 1, device=device, dtype=dtype)


def _discount_factor(rate: Tensor, time_index: Tensor) -> Tensor:
    return torch.clamp((1 + rate) ** time_index, min=EPSILON)


def _short_rate_tree(initial_rate: Tensor, periods: Tensor, rate_volatility: Tensor) -> list[Tensor]:
    count = max(1, int(periods.item()))
    dtype = initial_rate.dtype
    device = initial_rate.device
    sigma = torch.clamp(rate_volatility.to(device=device, dtype=dtype), min=0.0)
    if sigma.item() == 0:
        return [torch.full((n + 1,), initial_rate.item(), device=device, dtype=dtype) for n in range(count)]

    up = torch.exp(sigma)
    tree: list[Tensor] = []
    for n in range(count):
        exponents = torch.arange(-n, n + 1, 2, device=device, dtype=dtype)
        tree.append(torch.clamp(initial_rate * (up**exponents), min=0.0))
    return tree


def _embedded_option_bond_lattice(
    face_value: Tensor,
    coupon_rate: Tensor,
    rate: Tensor,
    periods: Tensor,
    *,
    exercise_price: Tensor | None = None,
    exercise_period: Tensor | None = None,
    exercise_kind: str | None = None,
    rate_volatility: Tensor | None = None,
) -> Tensor:
    period_count = max(1, int(periods.item()))
    dtype = face_value.dtype
    device = face_value.device
    initial_rate = rate.to(device=device, dtype=dtype)
    sigma_r = torch.zeros((), device=device, dtype=dtype) if rate_volatility is None else rate_volatility.to(device=device, dtype=dtype)
    rate_tree = _short_rate_tree(initial_rate, periods.to(device=device), sigma_r)
    coupon = face_value * coupon_rate.to(device=device, dtype=dtype)
    continuation = torch.full((period_count + 1,), face_value.item(), device=device, dtype=dtype)

    exercise_index = None if exercise_period is None else max(1, min(int(exercise_period.item()), period_count))
    exercise_value = None if exercise_price is None else exercise_price.to(device=device, dtype=dtype)

    for step in range(period_count - 1, -1, -1):
        expected_next = 0.5 * (continuation[:-1] + continuation[1:])
        node_values = (coupon + expected_next) / torch.clamp(1.0 + rate_tree[step], min=EPSILON)
        if exercise_index is not None and exercise_value is not None and step + 1 == exercise_index:
            if exercise_kind == "call":
                node_values = torch.minimum(node_values, exercise_value)
            elif exercise_kind == "put":
                node_values = torch.maximum(node_values, exercise_value)
        continuation = node_values
    return continuation[0]


def zero_coupon_bond(face_value: Tensor, rate: Tensor, maturity: Tensor) -> Tensor:
    return face_value / _discount_factor(rate, maturity)


def coupon_bond(face_value: Tensor, coupon_rate: Tensor, rate: Tensor, periods: Tensor) -> Tensor:
    period_grid = _period_grid(periods, device=face_value.device, dtype=face_value.dtype)
    coupon_cashflow = face_value * coupon_rate
    discount_curve = _discount_factor(rate, period_grid)
    discounted_coupons = coupon_cashflow / discount_curve
    discounted_principal = face_value / _discount_factor(rate, periods.to(face_value.dtype))
    return discounted_coupons.sum() + discounted_principal


def callable_bond_lattice(
    face_value: Tensor,
    coupon_rate: Tensor,
    rate: Tensor,
    periods: Tensor,
    call_price: Tensor,
    call_period: Tensor,
    rate_volatility: Tensor | None = None,
) -> Tensor:
    lattice_value = _embedded_option_bond_lattice(
        face_value,
        coupon_rate,
        rate,
        periods,
        exercise_price=call_price,
        exercise_period=call_period,
        exercise_kind="call",
        rate_volatility=rate_volatility,
    )
    return torch.minimum(lattice_value, coupon_bond(face_value, coupon_rate, rate, periods))


def callable_bond(face_value: Tensor, coupon_rate: Tensor, rate: Tensor, periods: Tensor, call_price: Tensor, call_period: Tensor) -> Tensor:
    return callable_bond_lattice(face_value, coupon_rate, rate, periods, call_price, call_period)


def putable_bond_lattice(
    face_value: Tensor,
    coupon_rate: Tensor,
    rate: Tensor,
    periods: Tensor,
    put_price: Tensor,
    put_period: Tensor,
    rate_volatility: Tensor | None = None,
) -> Tensor:
    lattice_value = _embedded_option_bond_lattice(
        face_value,
        coupon_rate,
        rate,
        periods,
        exercise_price=put_price,
        exercise_period=put_period,
        exercise_kind="put",
        rate_volatility=rate_volatility,
    )
    return torch.maximum(lattice_value, coupon_bond(face_value, coupon_rate, rate, periods))


def putable_bond(face_value: Tensor, coupon_rate: Tensor, rate: Tensor, periods: Tensor, put_price: Tensor, put_period: Tensor) -> Tensor:
    return putable_bond_lattice(face_value, coupon_rate, rate, periods, put_price, put_period)


def convertible_bond(face_value: Tensor, coupon_rate: Tensor, rate: Tensor, periods: Tensor, conversion_ratio: Tensor, conversion_price: Tensor) -> Tensor:
    straight_value = coupon_bond(face_value, coupon_rate, rate, periods)
    conversion_value = conversion_ratio * conversion_price
    return torch.maximum(straight_value, conversion_value)


def convertible_bond_lattice(
    face_value: Tensor,
    coupon_rate: Tensor,
    rate: Tensor,
    periods: Tensor,
    conversion_ratio: Tensor,
    stock_price: Tensor,
    stock_volatility: Tensor,
    *,
    dividend_yield: Tensor | None = None,
    call_price: Tensor | None = None,
    call_period: Tensor | None = None,
    put_price: Tensor | None = None,
    put_period: Tensor | None = None,
) -> Tensor:
    period_count = max(1, int(periods.item()))
    dtype = face_value.dtype
    device = face_value.device

    coupon = face_value * coupon_rate.to(device=device, dtype=dtype)
    rate_tensor = rate.to(device=device, dtype=dtype)
    stock0 = stock_price.to(device=device, dtype=dtype)
    volatility = torch.clamp(stock_volatility.to(device=device, dtype=dtype), min=0.0)
    dividend = torch.zeros((), device=device, dtype=dtype) if dividend_yield is None else dividend_yield.to(device=device, dtype=dtype)

    dt = torch.tensor(1.0, device=device, dtype=dtype)
    call_index = None if call_period is None else max(1, min(int(call_period.item()), period_count))
    call_value = None if call_price is None else call_price.to(device=device, dtype=dtype)
    put_index = None if put_period is None else max(1, min(int(put_period.item()), period_count))
    put_value = None if put_price is None else put_price.to(device=device, dtype=dtype)

    if volatility.item() == 0:
        growth = torch.exp((rate_tensor - dividend) * dt)
        terminal_stock = stock0 * growth**period_count
        terminal_value = torch.maximum(face_value, conversion_ratio * terminal_stock)
        if call_value is not None and call_index == period_count:
            terminal_value = torch.minimum(terminal_value, call_value)
        if put_value is not None and put_index == period_count:
            terminal_value = torch.maximum(terminal_value, put_value)
        continuation = torch.full((period_count + 1,), terminal_value.item(), device=device, dtype=dtype)
        for step in range(period_count - 1, -1, -1):
            continuation = (coupon + continuation[:-1]) / torch.clamp(1.0 + rate_tensor, min=EPSILON)
            if call_value is not None and call_index is not None and step + 1 == call_index:
                continuation = torch.minimum(continuation, call_value)
            if put_value is not None and put_index is not None and step + 1 == put_index:
                continuation = torch.maximum(continuation, put_value)
        return continuation[0]

    up = torch.exp(volatility * torch.sqrt(dt))
    down = 1.0 / up
    growth = torch.exp((rate_tensor - dividend) * dt)
    probability = torch.clamp((growth - down) / torch.clamp(up - down, min=EPSILON), min=0.0, max=1.0)

    terminal_stock = stock0 * (up ** torch.arange(period_count, -1, -1, device=device, dtype=dtype)) * (down ** torch.arange(0, period_count + 1, device=device, dtype=dtype))
    continuation = torch.maximum(face_value * torch.ones_like(terminal_stock), conversion_ratio * terminal_stock)
    if call_value is not None and call_index == period_count:
        continuation = torch.minimum(continuation, call_value)
    if put_value is not None and put_index == period_count:
        continuation = torch.maximum(continuation, put_value)

    for step in range(period_count - 1, -1, -1):
        stock_nodes = stock0 * (up ** torch.arange(step, -1, -1, device=device, dtype=dtype)) * (down ** torch.arange(0, step + 1, device=device, dtype=dtype))
        expected_next = probability * continuation[:-1] + (1.0 - probability) * continuation[1:]
        hold_value = (coupon + expected_next) / torch.clamp(1.0 + rate_tensor, min=EPSILON)
        conversion_value = conversion_ratio * stock_nodes
        node_value = torch.maximum(hold_value, conversion_value)
        if call_value is not None and call_index is not None and step + 1 == call_index:
            node_value = torch.minimum(node_value, call_value)
        if put_value is not None and put_index is not None and step + 1 == put_index:
            node_value = torch.maximum(node_value, put_value)
        continuation = node_value

    return continuation[0]


def callable_convertible_bond_lattice(
    face_value: Tensor,
    coupon_rate: Tensor,
    rate: Tensor,
    periods: Tensor,
    conversion_ratio: Tensor,
    stock_price: Tensor,
    stock_volatility: Tensor,
    call_price: Tensor,
    call_period: Tensor,
    *,
    dividend_yield: Tensor | None = None,
) -> Tensor:
    return convertible_bond_lattice(
        face_value,
        coupon_rate,
        rate,
        periods,
        conversion_ratio,
        stock_price,
        stock_volatility,
        dividend_yield=dividend_yield,
        call_price=call_price,
        call_period=call_period,
    )


def putable_convertible_bond_lattice(
    face_value: Tensor,
    coupon_rate: Tensor,
    rate: Tensor,
    periods: Tensor,
    conversion_ratio: Tensor,
    stock_price: Tensor,
    stock_volatility: Tensor,
    put_price: Tensor,
    put_period: Tensor,
    *,
    dividend_yield: Tensor | None = None,
) -> Tensor:
    return convertible_bond_lattice(
        face_value,
        coupon_rate,
        rate,
        periods,
        conversion_ratio,
        stock_price,
        stock_volatility,
        dividend_yield=dividend_yield,
        put_price=put_price,
        put_period=put_period,
    )


def callable_putable_convertible_bond_lattice(
    face_value: Tensor,
    coupon_rate: Tensor,
    rate: Tensor,
    periods: Tensor,
    conversion_ratio: Tensor,
    stock_price: Tensor,
    stock_volatility: Tensor,
    call_price: Tensor,
    call_period: Tensor,
    put_price: Tensor,
    put_period: Tensor,
    *,
    dividend_yield: Tensor | None = None,
) -> Tensor:
    return convertible_bond_lattice(
        face_value,
        coupon_rate,
        rate,
        periods,
        conversion_ratio,
        stock_price,
        stock_volatility,
        dividend_yield=dividend_yield,
        call_price=call_price,
        call_period=call_period,
        put_price=put_price,
        put_period=put_period,
    )


def stochastic_rate_bond(face_value: Tensor, coupon_rate: Tensor, rate: Tensor, periods: Tensor) -> Tensor:
    coupon = face_value * coupon_rate
    period_count = max(1, int(periods.item()))
    discount_factors = torch.cumprod(1 + rate[:period_count], dim=0)
    discounted_coupons = coupon / discount_factors
    principal = face_value / discount_factors[-1]
    return discounted_coupons.sum() + principal


__all__ = [
    "callable_bond",
    "callable_bond_lattice",
    "callable_convertible_bond_lattice",
    "convertible_bond",
    "convertible_bond_lattice",
    "putable_convertible_bond_lattice",
    "callable_putable_convertible_bond_lattice",
    "coupon_bond",
    "putable_bond",
    "putable_bond_lattice",
    "stochastic_rate_bond",
    "zero_coupon_bond",
]
