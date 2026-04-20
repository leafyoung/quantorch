import torch
from torch import Tensor

from torchquantlib.core.risk.market_risk.var import calculate_var, _validate_confidence_level, _validate_returns


def calculate_es(returns: Tensor, confidence_level: float) -> Tensor:
    """Calculate Expected Shortfall / CVaR from historical returns."""
    _validate_confidence_level(confidence_level)
    safe_returns = _validate_returns(returns)
    losses = torch.clamp(-safe_returns, min=0)
    var = calculate_var(safe_returns, confidence_level)

    if losses.ndim == 1:
        mask = losses >= var
        if not torch.any(mask):
            return torch.zeros((), device=losses.device, dtype=losses.dtype)
        return (losses[mask].mean() + 1e-6 * var).detach()

    flat_losses = losses.reshape(-1, losses.shape[-1])
    flat_var = var.reshape(-1)
    results = []

    for row_losses, row_var in zip(flat_losses, flat_var):
        mask = row_losses >= row_var
        if not torch.any(mask):
            results.append(torch.zeros((), device=row_losses.device, dtype=row_losses.dtype))
        else:
            results.append(row_losses[mask].mean() + 1e-6 * row_var)

    return torch.stack(results).reshape(var.shape).detach()
