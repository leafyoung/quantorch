import torch
from torch import Tensor

from torchquantlib.core.asset_pricing.option.utils import calculate_binomial_tree_params


ZERO = torch.tensor(0.0)


def american_option(
    option_type: str,
    spot: Tensor,
    strike: Tensor,
    expiry: Tensor,
    volatility: Tensor,
    rate: Tensor,
    steps: int,
) -> Tensor:
    """Price an American option with a binomial tree.

    This public API matches the existing tests, examples, and docs: callers pass a
    single ``steps`` argument that controls the binomial tree resolution.
    """
    if option_type not in {"call", "put"}:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    if int(steps) <= 0:
        raise ValueError("steps must be a positive integer.")
    if volatility.item() <= 0:
        raise RuntimeError("volatility must be positive for american_option.")

    steps = int(steps)
    dt, u, d, p = calculate_binomial_tree_params(expiry, volatility, rate, steps)

    indices = torch.arange(steps + 1, dtype=torch.float32, device=spot.device)
    i, j = torch.meshgrid(indices, indices, indexing="ij")
    mask = j <= i
    price_tree = spot * (u ** (i - j)) * (d ** j) * mask
    value_tree = torch.zeros_like(price_tree)

    if option_type == "call":
        value_tree[:, steps] = torch.maximum(price_tree[:, steps] - strike, ZERO.to(spot.device))
    else:
        value_tree[:, steps] = torch.maximum(strike - price_tree[:, steps], ZERO.to(spot.device))

    discount_factor = torch.exp(-rate * dt)

    for step in range(steps - 1, -1, -1):
        continuation_value = (
            p * value_tree[: step + 1, step + 1] + (1 - p) * value_tree[1 : step + 2, step + 1]
        ) * discount_factor

        if option_type == "call":
            exercise_value = torch.maximum(price_tree[: step + 1, step] - strike, ZERO.to(spot.device))
        else:
            exercise_value = torch.maximum(strike - price_tree[: step + 1, step], ZERO.to(spot.device))

        value_tree[: step + 1, step] = torch.maximum(exercise_value, continuation_value)

    return value_tree[0, 0]


__all__ = ["american_option"]
