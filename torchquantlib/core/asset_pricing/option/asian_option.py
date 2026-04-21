from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from torchquantlib.core.asset_pricing.option.exotics import asian_option as monte_carlo_asian_option

EPSILON = 1e-8


def _to_tensor(value, device: torch.device | None = None) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.float32)
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    return tensor.to(device) if device is not None else tensor


def _normal_cdf(x: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _sampling_times(expiry: Tensor, steps: int) -> Tensor:
    return expiry * torch.arange(1, steps + 1, device=expiry.device, dtype=expiry.dtype) / steps


def _average_forward_mean(spot: Tensor, expiry: Tensor, rate: Tensor, steps: int) -> Tensor:
    times = _sampling_times(expiry, steps)
    return spot * torch.exp(rate * times).mean()


def _discounted_intrinsic(option_type: str, average_level: Tensor, strike: Tensor, expiry: Tensor, rate: Tensor) -> Tensor:
    payoff = torch.relu(average_level - strike) if option_type == "call" else torch.relu(strike - average_level)
    return torch.exp(-rate * expiry) * payoff


def _lognormal_option_price(option_type: str, mean_level: Tensor, second_moment: Tensor, strike: Tensor, expiry: Tensor, rate: Tensor) -> Tensor:
    mean_level = torch.clamp(mean_level, min=EPSILON)
    second_moment = torch.clamp(second_moment, min=mean_level**2)
    log_variance = torch.log(torch.clamp(second_moment / torch.clamp(mean_level**2, min=EPSILON), min=1.0))

    if log_variance.item() <= EPSILON:
        return _discounted_intrinsic(option_type, mean_level, strike, expiry, rate)

    sqrt_log_variance = torch.sqrt(log_variance)
    d1 = (torch.log(mean_level / strike) + 0.5 * log_variance) / sqrt_log_variance
    d2 = d1 - sqrt_log_variance
    discount_factor = torch.exp(-rate * expiry)

    if option_type == "call":
        return discount_factor * (mean_level * _normal_cdf(d1) - strike * _normal_cdf(d2))
    return discount_factor * (strike * _normal_cdf(-d2) - mean_level * _normal_cdf(-d1))


def _geometric_asian_price(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, steps: int) -> Tensor:
    times = _sampling_times(expiry, steps)
    log_spot_mean = torch.log(torch.clamp(spot, min=EPSILON)) + (rate - 0.5 * volatility**2) * times.mean()
    covariance_matrix = torch.minimum(times[:, None], times[None, :])
    log_variance = volatility**2 * covariance_matrix.mean()
    second_moment = torch.exp(2.0 * log_spot_mean + 2.0 * log_variance)
    mean_level = torch.exp(log_spot_mean + 0.5 * log_variance)
    return _lognormal_option_price(option_type, mean_level, second_moment, strike, expiry, rate)


def _arithmetic_asian_lognormal_price(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, steps: int) -> Tensor:
    times = _sampling_times(expiry, steps)
    mean_level = _average_forward_mean(spot, expiry, rate, steps)
    pairwise_min = torch.minimum(times[:, None], times[None, :])
    pairwise_growth = torch.exp(rate * (times[:, None] + times[None, :]) + volatility**2 * pairwise_min)
    second_moment = spot**2 * pairwise_growth.mean()
    return _lognormal_option_price(option_type, mean_level, second_moment, strike, expiry, rate)


@dataclass(frozen=True)
class AsianOption:
    underlying: float | Tensor
    strike: float | Tensor
    expiry: float | Tensor
    rf_rate: float | Tensor
    volatility: float | Tensor
    option_type: str = "call"
    averaging_type: str = "arithmetic"
    averaging_period: float | Tensor | None = None
    steps: int = 252

    def _normalized_inputs(self):
        spot = _to_tensor(self.underlying)
        strike = torch.clamp(_to_tensor(self.strike, device=spot.device), min=EPSILON)
        expiry = _to_tensor(self.expiry, device=spot.device)
        rate = _to_tensor(self.rf_rate, device=spot.device)
        volatility = _to_tensor(self.volatility, device=spot.device)
        return spot, strike, expiry, volatility, rate

    def price(self, method: str = "monte_carlo", num_simulations: int = 10000, num_steps: int | None = None, seed: int = 42) -> Tensor:
        if self.option_type not in {"call", "put"}:
            raise ValueError("Option type must be either 'call' or 'put'.")
        if self.averaging_type not in {"arithmetic", "geometric"}:
            raise ValueError("average_type must be either 'arithmetic' or 'geometric'.")

        steps = self.steps if num_steps is None else int(num_steps)
        if method == "monte_carlo":
            return monte_carlo_price(self, num_simulations=num_simulations, num_steps=steps, seed=seed)
        if method == "analytical_approximation":
            return analytical_approximation_price(self, steps=steps)
        raise ValueError("method must be 'monte_carlo' or 'analytical_approximation'.")

    def _finite_difference_greek(self, field_name: str, bump_size: float, *, sign: float = 1.0) -> float:
        base_kwargs = {
            "underlying": self.underlying,
            "strike": self.strike,
            "expiry": self.expiry,
            "rf_rate": self.rf_rate,
            "volatility": self.volatility,
            "option_type": self.option_type,
            "averaging_type": self.averaging_type,
            "averaging_period": self.averaging_period,
            "steps": self.steps,
        }
        base_value = base_kwargs[field_name]

        bumped_up = AsianOption(**{**base_kwargs, field_name: _to_tensor(base_value) + bump_size})
        bumped_down = AsianOption(**{**base_kwargs, field_name: _to_tensor(base_value) - bump_size})

        price_up = bumped_up.price(method="analytical_approximation")
        price_down = bumped_down.price(method="analytical_approximation")
        greek = sign * (price_up - price_down) / (2.0 * bump_size)
        return float(greek.item())

    def delta(self) -> float:
        return self._finite_difference_greek("underlying", bump_size=1e-2)

    def gamma(self) -> float:
        base_kwargs = {
            "underlying": self.underlying,
            "strike": self.strike,
            "expiry": self.expiry,
            "rf_rate": self.rf_rate,
            "volatility": self.volatility,
            "option_type": self.option_type,
            "averaging_type": self.averaging_type,
            "averaging_period": self.averaging_period,
            "steps": self.steps,
        }
        bump = 1e-2
        price_center = self.price(method="analytical_approximation")
        price_up = AsianOption(**{**base_kwargs, "underlying": _to_tensor(self.underlying) + bump}).price(method="analytical_approximation")
        price_down = AsianOption(**{**base_kwargs, "underlying": _to_tensor(self.underlying) - bump}).price(method="analytical_approximation")
        gamma = (price_up - 2.0 * price_center + price_down) / (bump**2)
        return float(gamma.item())

    def theta(self) -> float:
        return self._finite_difference_greek("expiry", bump_size=1e-4, sign=-1.0)

    def vega(self) -> float:
        return self._finite_difference_greek("volatility", bump_size=1e-4)

    def rho(self) -> float:
        return self._finite_difference_greek("rf_rate", bump_size=1e-4)


def monte_carlo_price(option: AsianOption, num_simulations: int = 10000, num_steps: int | None = None, seed: int = 42) -> Tensor:
    spot, strike, expiry, volatility, rate = option._normalized_inputs()
    steps = option.steps if num_steps is None else int(num_steps)

    if volatility.item() <= 0:
        average_level = _average_forward_mean(spot, expiry, rate, steps)
        return _discounted_intrinsic(option.option_type, average_level, strike, expiry, rate)

    torch.manual_seed(seed)
    return monte_carlo_asian_option(
        option_type=option.option_type,
        spot=spot,
        strike=strike,
        expiry=expiry,
        volatility=volatility,
        rate=rate,
        num_paths=int(num_simulations),
        num_steps=steps,
        average_type=option.averaging_type,
    )


def analytical_approximation_price(option: AsianOption, steps: int | None = None) -> Tensor:
    spot, strike, expiry, volatility, rate = option._normalized_inputs()
    steps = option.steps if steps is None else int(steps)

    if volatility.item() <= 0:
        average_level = _average_forward_mean(spot, expiry, rate, steps)
        return _discounted_intrinsic(option.option_type, average_level, strike, expiry, rate)

    if option.averaging_type == "geometric":
        return _geometric_asian_price(option.option_type, spot, strike, expiry, volatility, rate, steps)
    return _arithmetic_asian_lognormal_price(option.option_type, spot, strike, expiry, volatility, rate, steps)


def asian_option(
    option_type: str,
    spot: Tensor,
    strike: Tensor,
    expiry: Tensor,
    volatility: Tensor,
    rate: Tensor,
    steps: int,
    average_type: str = "arithmetic",
    num_paths: int | None = None,
    seed: int = 42,
) -> Tensor:
    if option_type not in {"call", "put"}:
        raise ValueError("Option type must be either 'call' or 'put'.")
    option = AsianOption(
        underlying=spot,
        strike=strike,
        expiry=expiry,
        rf_rate=rate,
        volatility=volatility,
        option_type=option_type,
        averaging_type=average_type,
        steps=int(steps),
    )

    if num_paths is None:
        return analytical_approximation_price(option, steps=int(steps))
    return monte_carlo_price(option, num_simulations=int(num_paths), num_steps=int(steps), seed=seed)


__all__ = [
    "AsianOption",
    "analytical_approximation_price",
    "asian_option",
    "monte_carlo_price",
]
