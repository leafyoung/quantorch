import torch
from torch import Tensor

from torchquantlib.core.asset_pricing.option.utils import price_binomial_tree


def _to_batch_tensor(value) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.as_tensor(value, dtype=torch.float32)


def binomial_tree(option_type: str, option_style: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, steps: int) -> Tensor:
    """Price European or American options with a binomial tree.

    Supports scalar tensors and simple 1D batch inputs by evaluating each batch
    element independently.
    """
    spot = _to_batch_tensor(spot)
    strike = _to_batch_tensor(strike).to(device=spot.device, dtype=spot.dtype)
    expiry = _to_batch_tensor(expiry).to(device=spot.device, dtype=spot.dtype)
    volatility = _to_batch_tensor(volatility).to(device=spot.device, dtype=spot.dtype)
    rate = _to_batch_tensor(rate).to(device=spot.device, dtype=spot.dtype)

    if spot.ndim == 0:
        return price_binomial_tree(option_type, option_style, spot, strike, expiry, volatility, rate, steps)

    batch_size = int(spot.shape[0])
    outputs = []
    for index in range(batch_size):
        outputs.append(
            price_binomial_tree(
                option_type,
                option_style,
                spot[index],
                strike[index],
                expiry[index],
                volatility[index],
                rate[index],
                steps,
            )
        )
    return torch.stack(outputs)
