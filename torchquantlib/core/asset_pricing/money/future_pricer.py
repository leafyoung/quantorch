import torch
from torch import Tensor


def future_pricer(spot: Tensor, rate: Tensor, expiry: Tensor, foreign_rate: Tensor | None = None) -> Tensor:
    """Calculate the theoretical futures price.

    Public test coverage expects ``future_pricer(spot, rate, expiry)``. An optional
    ``foreign_rate`` is still supported for cost-of-carry style use cases.
    """
    if foreign_rate is None:
        carry_rate = rate
    else:
        carry_rate = rate - foreign_rate
    return spot * torch.exp(carry_rate * expiry)
