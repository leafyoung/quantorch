import torch
from torch import Tensor


def _validate_returns(returns: Tensor) -> Tensor:
    if not isinstance(returns, Tensor):
        raise TypeError("returns must be a torch.Tensor.")
    if returns.numel() == 0:
        raise RuntimeError("returns must not be empty.")
    return returns.detach()


def _validate_confidence_level(confidence_level: float) -> None:
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")


def calculate_var(returns: Tensor, confidence_level: float) -> Tensor:
    """Calculate historical-simulation VaR from return series.

    The implementation converts returns into non-negative losses and then applies
    a quantile on the last dimension. Positive-return-only samples therefore map
    to zero loss and zero VaR, which matches the test suite's expected behavior.
    """
    _validate_confidence_level(confidence_level)
    safe_returns = _validate_returns(returns)

    losses = torch.clamp(-safe_returns, min=0)
    var = torch.quantile(losses, confidence_level, dim=-1)
    return var.detach()
