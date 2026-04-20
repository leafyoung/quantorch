from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from torchquantlib.core.asset_pricing.option.black_scholes_merton import black_scholes_merton
from torchquantlib.core.asset_pricing.option.exotics import asian_option as monte_carlo_asian_option

EPSILON = 1e-8


def _to_tensor(value, device: torch.device | None = None) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.float32)
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    return tensor.to(device) if device is not None else tensor


def _discounted_intrinsic_from_average(
    option_type: str,
    spot: Tensor,
    strike: Tensor,
    expiry: Tensor,
    rate: Tensor,
) -> Tensor:
    if torch.abs(rate) < EPSILON:
        expected_average = spot
    else:
        expected_average = spot * (torch.exp(rate * expiry) - 1.0) / (rate * expiry)

    payoff = torch.relu(expected_average - strike) if option_type == "call" else torch.relu(strike - expected_average)
    return payoff * torch.exp(-rate * expiry)


def _effective_asian_volatility(volatility: Tensor, steps: int) -> Tensor:
    discrete_adjustment = torch.sqrt(torch.tensor((2 * steps + 1) / (6 * (steps + 1)), device=volatility.device))
    return torch.clamp(volatility * discrete_adjustment, min=EPSILON)


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
        strike = _to_tensor(self.strike, device=spot.device)
        expiry = _to_tensor(self.expiry, device=spot.device)
        rate = _to_tensor(self.rf_rate, device=spot.device)
        volatility = _to_tensor(self.volatility, device=spot.device)
        return spot, strike, expiry, volatility, rate

    def price(self, method: str = "monte_carlo", num_simulations: int = 10000, num_steps: int | None = None, seed: int = 42) -> Tensor:
        if self.option_type not in {"call", "put"}:
            raise ValueError("Option type must be either 'call' or 'put'.")
        if self.averaging_type not in {"arithmetic", "geometric"}:
            raise ValueError("average_type must be either 'arithmetic' or 'geometric'.")

        spot, strike, expiry, volatility, rate = self._normalized_inputs()
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
        return _discounted_intrinsic_from_average(option.option_type, spot, strike, expiry, rate)

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
        return _discounted_intrinsic_from_average(option.option_type, spot, strike, expiry, rate)

    effective_volatility = _effective_asian_volatility(volatility, steps)
    return black_scholes_merton(
        option_type=option.option_type,
        option_style="european",
        spot=spot,
        strike=strike,
        expiry=expiry,
        volatility=effective_volatility,
        rate=rate,
        dividend=torch.zeros_like(rate),
    )


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
