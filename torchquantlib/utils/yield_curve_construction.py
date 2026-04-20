import torch
from torch import Tensor


def bootstrap_yield_curve(cash_flows: Tensor, prices: Tensor) -> Tensor:
    """Construct simple yields from cash-flow totals and market prices.

    The helper supports both 2D inputs ``(n_instruments, n_cashflows)`` and batched
    3D inputs ``(batch, n_instruments, n_cashflows)`` with matching price shapes.
    """
    if not isinstance(cash_flows, Tensor) or not isinstance(prices, Tensor):
        raise TypeError("cash_flows and prices must be torch.Tensor inputs.")
    if torch.any(prices <= 0):
        raise RuntimeError("prices must be strictly positive.")
    if cash_flows.shape[:-1] != prices.shape:
        raise RuntimeError("cash_flows and prices must agree on instrument dimensions.")

    cash_flow_totals = cash_flows.sum(dim=-1)
    num_instruments = cash_flow_totals.shape[-1]
    periods = torch.arange(1, num_instruments + 1, device=prices.device, dtype=prices.dtype)

    view_shape = [1] * cash_flow_totals.ndim
    view_shape[-1] = num_instruments
    exponents = (1.0 / periods).reshape(view_shape)

    yields = torch.pow(cash_flow_totals / prices, exponents) - 1.0
    return yields


def nelson_siegel_yield_curve(tau: Tensor, beta0: Tensor, beta1: Tensor, beta2: Tensor) -> Tensor:
    """Construct a Nelson-Siegel yield curve with lambda fixed at 1."""
    if not all(isinstance(value, Tensor) for value in (tau, beta0, beta1, beta2)):
        raise TypeError("tau, beta0, beta1, and beta2 must be torch.Tensor inputs.")
    if torch.any(tau <= 0):
        raise RuntimeError("tau must be strictly positive.")

    tau = tau.to(dtype=torch.float32)
    beta0 = beta0.to(device=tau.device, dtype=tau.dtype)
    beta1 = beta1.to(device=tau.device, dtype=tau.dtype)
    beta2 = beta2.to(device=tau.device, dtype=tau.dtype)

    exp_term = torch.exp(-tau)
    factor1 = (1 - exp_term) / tau
    factor2 = factor1 - exp_term

    while beta0.ndim < tau.ndim:
        beta0 = beta0.unsqueeze(-1)
        beta1 = beta1.unsqueeze(-1)
        beta2 = beta2.unsqueeze(-1)

    return beta0 + beta1 * factor1 + beta2 * factor2
