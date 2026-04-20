import torch
from torch import Tensor


def _validate_non_negative(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if torch.any(value < 0):
        raise RuntimeError(f"{name} must be non-negative.")


def _validate_recovery_rate(recovery_rate: Tensor) -> None:
    _validate_non_negative("recovery_rate", recovery_rate)
    if torch.any(recovery_rate > 1):
        raise RuntimeError("recovery_rate must be between 0 and 1.")


def calculate_cva(exposure: Tensor, default_prob: Tensor, recovery_rate: Tensor) -> Tensor:
    _validate_non_negative("exposure", exposure)
    _validate_non_negative("default_prob", default_prob)
    _validate_recovery_rate(recovery_rate)
    return exposure * (1 - recovery_rate) * default_prob


def calculate_dva(exposure: Tensor, default_prob: Tensor, recovery_rate: Tensor) -> Tensor:
    _validate_non_negative("exposure", exposure)
    _validate_non_negative("default_prob", default_prob)
    _validate_recovery_rate(recovery_rate)
    return exposure * (1 - recovery_rate) * default_prob


def calculate_fva(exposure: Tensor, funding_spread: Tensor, maturity: Tensor) -> Tensor:
    _validate_non_negative("exposure", exposure)
    _validate_non_negative("funding_spread", funding_spread)
    _validate_non_negative("maturity", maturity)
    return exposure * funding_spread * maturity


def calculate_mva(exposure: Tensor, funding_cost: Tensor, maturity: Tensor) -> Tensor:
    _validate_non_negative("exposure", exposure)
    _validate_non_negative("funding_cost", funding_cost)
    _validate_non_negative("maturity", maturity)
    return exposure * funding_cost * maturity
