import torch
from torch import Tensor


def zero_coupon_bond(face_value: Tensor, rate: Tensor, maturity: Tensor) -> Tensor:
    return face_value / (1 + rate) ** maturity


def coupon_bond(face_value: Tensor, coupon_rate: Tensor, rate: Tensor, periods: Tensor) -> Tensor:
    coupon_total = face_value * coupon_rate * periods
    return face_value + coupon_total / (1 + rate)


def callable_bond(face_value: Tensor, coupon_rate: Tensor, rate: Tensor, periods: Tensor, call_price: Tensor, call_period: Tensor) -> Tensor:
    del call_period
    return torch.minimum(coupon_bond(face_value, coupon_rate, rate, periods), call_price)


def putable_bond(face_value: Tensor, coupon_rate: Tensor, rate: Tensor, periods: Tensor, put_price: Tensor, put_period: Tensor) -> Tensor:
    del put_period
    return torch.maximum(coupon_bond(face_value, coupon_rate, rate, periods), put_price)


def convertible_bond(face_value: Tensor, coupon_rate: Tensor, rate: Tensor, periods: Tensor, conversion_ratio: Tensor, conversion_price: Tensor) -> Tensor:
    straight_value = coupon_bond(face_value, coupon_rate, rate, periods)
    conversion_value = conversion_ratio * conversion_price
    return torch.minimum(straight_value, conversion_value)


def stochastic_rate_bond(face_value: Tensor, coupon_rate: Tensor, rate: Tensor, periods: Tensor) -> Tensor:
    coupon = face_value * coupon_rate
    period_indices = torch.arange(1, int(periods.item()) + 1, device=rate.device, dtype=rate.dtype)
    discounted_coupons = coupon / (1 + rate[: int(periods.item())]) ** period_indices
    principal = face_value / (1 + rate[int(periods.item()) - 1]) ** periods
    return discounted_coupons.sum() + principal
