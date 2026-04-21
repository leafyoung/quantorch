from __future__ import annotations

import torch
from torch import Tensor


EPSILON = 1e-8


def _period_grid(periods: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
    count = max(1, int(periods.item()))
    return torch.arange(1, count + 1, device=device, dtype=dtype)


def _discount_factor(rate: Tensor, time_index: Tensor) -> Tensor:
    return torch.clamp((1 + rate) ** time_index, min=EPSILON)


def zero_coupon_bond(face_value: Tensor, rate: Tensor, maturity: Tensor) -> Tensor:
    return face_value / _discount_factor(rate, maturity)


def coupon_bond(face_value: Tensor, coupon_rate: Tensor, rate: Tensor, periods: Tensor) -> Tensor:
    period_grid = _period_grid(periods, device=face_value.device, dtype=face_value.dtype)
    coupon_cashflow = face_value * coupon_rate
    discount_curve = _discount_factor(rate, period_grid)
    discounted_coupons = coupon_cashflow / discount_curve
    discounted_principal = face_value / _discount_factor(rate, periods.to(face_value.dtype))
    return discounted_coupons.sum() + discounted_principal


def callable_bond(face_value: Tensor, coupon_rate: Tensor, rate: Tensor, periods: Tensor, call_price: Tensor, call_period: Tensor) -> Tensor:
    straight_value = coupon_bond(face_value, coupon_rate, rate, periods)
    call_period_int = max(1, min(int(call_period.item()), int(periods.item())))
    early_periods = torch.arange(1, call_period_int, device=face_value.device, dtype=face_value.dtype)
    coupon_cashflow = face_value * coupon_rate
    early_coupon_value = torch.zeros((), device=face_value.device, dtype=face_value.dtype)
    if len(early_periods) > 0:
        early_coupon_value = (coupon_cashflow / _discount_factor(rate, early_periods)).sum()
    exercise_value = early_coupon_value + (coupon_cashflow + call_price) / _discount_factor(rate, torch.tensor(float(call_period_int), device=face_value.device, dtype=face_value.dtype))
    return torch.minimum(straight_value, exercise_value)


def putable_bond(face_value: Tensor, coupon_rate: Tensor, rate: Tensor, periods: Tensor, put_price: Tensor, put_period: Tensor) -> Tensor:
    straight_value = coupon_bond(face_value, coupon_rate, rate, periods)
    put_period_int = max(1, min(int(put_period.item()), int(periods.item())))
    early_periods = torch.arange(1, put_period_int, device=face_value.device, dtype=face_value.dtype)
    coupon_cashflow = face_value * coupon_rate
    early_coupon_value = torch.zeros((), device=face_value.device, dtype=face_value.dtype)
    if len(early_periods) > 0:
        early_coupon_value = (coupon_cashflow / _discount_factor(rate, early_periods)).sum()
    exercise_value = early_coupon_value + (coupon_cashflow + put_price) / _discount_factor(rate, torch.tensor(float(put_period_int), device=face_value.device, dtype=face_value.dtype))
    return torch.maximum(straight_value, exercise_value)


def convertible_bond(face_value: Tensor, coupon_rate: Tensor, rate: Tensor, periods: Tensor, conversion_ratio: Tensor, conversion_price: Tensor) -> Tensor:
    straight_value = coupon_bond(face_value, coupon_rate, rate, periods)
    conversion_value = conversion_ratio * conversion_price
    return torch.maximum(straight_value, conversion_value)


def stochastic_rate_bond(face_value: Tensor, coupon_rate: Tensor, rate: Tensor, periods: Tensor) -> Tensor:
    coupon = face_value * coupon_rate
    period_count = max(1, int(periods.item()))
    period_indices = torch.arange(1, period_count + 1, device=rate.device, dtype=rate.dtype)
    discount_factors = torch.cumprod(1 + rate[:period_count], dim=0)
    discounted_coupons = coupon / discount_factors
    principal = face_value / discount_factors[-1]
    return discounted_coupons.sum() + principal
