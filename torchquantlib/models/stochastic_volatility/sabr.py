from __future__ import annotations

import math

import torch

from torchquantlib.models.stochastic_model import StochasticModel

EPS = 1e-8
ATM_TOL = 1e-7


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _normal_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def _black76_price(
    option_type: str,
    forward: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    volatility: torch.Tensor,
    rate: torch.Tensor,
) -> torch.Tensor:
    discount_factor = torch.exp(-rate * expiry)
    intrinsic_value = torch.maximum(
        forward - strike if option_type == "call" else strike - forward,
        torch.zeros_like(forward),
    )

    if expiry.item() <= 0 or volatility.item() <= 0:
        return discount_factor * intrinsic_value

    sqrt_expiry = torch.sqrt(expiry)
    volatility_term = volatility * sqrt_expiry
    d1 = (torch.log(forward / strike) + 0.5 * volatility**2 * expiry) / volatility_term
    d2 = d1 - volatility_term

    if option_type == "call":
        return discount_factor * (forward * _normal_cdf(d1) - strike * _normal_cdf(d2))
    return discount_factor * (strike * _normal_cdf(-d2) - forward * _normal_cdf(-d1))


def _bachelier_price(
    option_type: str,
    forward: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    volatility: torch.Tensor,
    rate: torch.Tensor,
) -> torch.Tensor:
    discount_factor = torch.exp(-rate * expiry)
    intrinsic_value = torch.maximum(
        forward - strike if option_type == "call" else strike - forward,
        torch.zeros_like(forward),
    )

    if expiry.item() <= 0 or volatility.item() <= 0:
        return discount_factor * intrinsic_value

    sqrt_expiry = torch.sqrt(expiry)
    denominator = torch.clamp(volatility * sqrt_expiry, min=EPS)
    d = (forward - strike) / denominator
    call_price = discount_factor * ((forward - strike) * _normal_cdf(d) + volatility * sqrt_expiry * _normal_pdf(d))
    if option_type == "call":
        return call_price
    return call_price - discount_factor * (forward - strike)


class SABR(StochasticModel):
    """SABR model with Hagan implied-volatility pricing.

    ``option_price`` supports both lognormal and normal Hagan asymptotic
    approximations, together with optional shifting for low or negative forward
    levels. ``simulate`` remains a simple Euler-style SABR sampler for terminal
    forwards to preserve the stochastic-model API used elsewhere in the repo.
    """

    parameter_constraints = {
        "alpha": {"min": 1.0001e-6},
        "beta": {"min": 0.0, "max": 1.0},
        "rho": {"min": -0.9989, "max": 0.9989},
        "nu": {"min": 1.0001e-6},
    }

    def __init__(self, alpha_init=0.2, beta_init=0.5, rho_init=0.0, nu_init=0.3, F0=100.0, shift_init=0.0):
        super().__init__(
            self.build_parameters(
                {
                    "alpha": alpha_init,
                    "beta": beta_init,
                    "rho": rho_init,
                    "nu": nu_init,
                }
            )
        )
        self.register_auxiliary_tensor("F0", F0)
        self.register_auxiliary_tensor("shift", shift_init)

    def _resolve_shift(self, shift) -> torch.Tensor:
        base_shift = self.shift if shift is None else shift
        return self.tensor(base_shift)

    def _base_forward_from_state(self, base_level, expiry, rate=None, dividend=0.0) -> torch.Tensor:
        forward = self.tensor(base_level if base_level is not None else self.F0)
        if rate is None:
            return forward
        rate_tensor = self.tensor(rate)
        dividend_tensor = self.tensor(dividend)
        return forward * torch.exp((rate_tensor - dividend_tensor) * expiry)

    def _shifted_forward_and_strike(self, forward, strike, shift=None, *, positive_required: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shift_tensor = self._resolve_shift(shift)
        shifted_forward = self.tensor(forward) + shift_tensor
        shifted_strike = self.tensor(strike) + shift_tensor
        if positive_required and (shifted_forward.item() <= 0 or shifted_strike.item() <= 0):
            raise ValueError("shifted forward and strike must be positive for lognormal SABR.")
        return shifted_forward, shifted_strike, shift_tensor

    def _atm_lognormal_implied_volatility(self, forward: torch.Tensor, expiry: torch.Tensor) -> torch.Tensor:
        alpha = self.constrained_param("alpha")
        beta = self.constrained_param("beta")
        rho = self.constrained_param("rho")
        nu = self.constrained_param("nu")

        forward = torch.clamp(forward, min=EPS)
        one_minus_beta = 1.0 - beta
        forward_power = forward ** one_minus_beta
        correction = (
            ((one_minus_beta**2) / 24.0) * (alpha**2 / torch.clamp(forward ** (2.0 * one_minus_beta), min=EPS))
            + (rho * beta * nu * alpha) / (4.0 * torch.clamp(forward_power, min=EPS))
            + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
        )
        return torch.clamp((alpha / torch.clamp(forward_power, min=EPS)) * (1.0 + correction * expiry), min=EPS)

    def _hagan_lognormal_implied_volatility(self, strike, expiry, *, forward: torch.Tensor | None = None, shift=None) -> torch.Tensor:
        expiry = self.tensor(expiry)
        if expiry.item() <= 0:
            return torch.zeros((), device=self.device)

        alpha = self.constrained_param("alpha")
        beta = self.constrained_param("beta")
        rho = self.constrained_param("rho")
        nu = self.constrained_param("nu")

        base_forward = self._base_forward_from_state(self.F0 if forward is None else forward, expiry)
        shifted_forward, shifted_strike, _ = self._shifted_forward_and_strike(base_forward, strike, shift, positive_required=True)
        shifted_forward = torch.clamp(shifted_forward, min=EPS)
        shifted_strike = torch.clamp(shifted_strike, min=EPS)

        log_fk = torch.log(shifted_forward / shifted_strike)
        if torch.abs(log_fk).item() < ATM_TOL:
            return self._atm_lognormal_implied_volatility(shifted_forward, expiry)

        one_minus_beta = 1.0 - beta
        fk_beta = torch.clamp((shifted_forward * shifted_strike) ** (0.5 * one_minus_beta), min=EPS)
        log_fk_sq = log_fk**2
        log_fk_4 = log_fk_sq**2

        z = (nu / torch.clamp(alpha, min=EPS)) * fk_beta * log_fk
        sqrt_term = torch.sqrt(torch.clamp(1.0 - 2.0 * rho * z + z**2, min=EPS))
        x_z = torch.log(torch.clamp((sqrt_term + z - rho) / (1.0 - rho), min=EPS))

        log_denominator = 1.0 + ((one_minus_beta**2) / 24.0) * log_fk_sq + ((one_minus_beta**4) / 1920.0) * log_fk_4
        prefactor = alpha / torch.clamp(fk_beta * log_denominator, min=EPS)
        correction = (
            ((one_minus_beta**2) / 24.0) * (alpha**2 / torch.clamp((shifted_forward * shifted_strike) ** one_minus_beta, min=EPS))
            + (rho * beta * nu * alpha) / (4.0 * torch.clamp(fk_beta, min=EPS))
            + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
        )

        ratio = torch.where(torch.abs(x_z) < ATM_TOL, torch.ones_like(z), z / x_z)
        return torch.clamp(prefactor * ratio * (1.0 + correction * expiry), min=EPS)

    def _atm_normal_implied_volatility(self, forward: torch.Tensor, expiry: torch.Tensor) -> torch.Tensor:
        alpha = self.constrained_param("alpha")
        beta = self.constrained_param("beta")
        rho = self.constrained_param("rho")
        nu = self.constrained_param("nu")

        forward = torch.clamp(forward, min=EPS)
        forward_beta = forward**beta
        one_minus_beta = 1.0 - beta
        correction = (
            -beta * (2.0 - beta) * alpha**2 / (24.0 * torch.clamp(forward ** (2.0 * one_minus_beta), min=EPS))
            + (rho * beta * nu * alpha) / (4.0 * torch.clamp(forward**one_minus_beta, min=EPS))
            + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
        )
        return torch.clamp(alpha * forward_beta * (1.0 + correction * expiry), min=EPS)

    def _hagan_normal_implied_volatility(self, strike, expiry, *, forward: torch.Tensor | None = None, shift=None) -> torch.Tensor:
        expiry = self.tensor(expiry)
        if expiry.item() <= 0:
            return torch.zeros((), device=self.device)

        alpha = self.constrained_param("alpha")
        beta = self.constrained_param("beta")
        rho = self.constrained_param("rho")
        nu = self.constrained_param("nu")

        base_forward = self._base_forward_from_state(self.F0 if forward is None else forward, expiry)
        shifted_forward, shifted_strike, _ = self._shifted_forward_and_strike(base_forward, strike, shift, positive_required=False)

        diff = shifted_forward - shifted_strike
        if torch.abs(diff).item() < ATM_TOL:
            return self._atm_normal_implied_volatility(torch.clamp(shifted_forward, min=EPS), expiry)

        fk_beta = torch.clamp((torch.clamp(shifted_forward * shifted_strike, min=EPS)) ** (0.5 * beta), min=EPS)
        zeta = (nu / torch.clamp(alpha, min=EPS)) * diff / fk_beta
        sqrt_term = torch.sqrt(torch.clamp(1.0 - 2.0 * rho * zeta + zeta**2, min=EPS))
        x_zeta = torch.log(torch.clamp((sqrt_term + zeta - rho) / (1.0 - rho), min=EPS))

        correction = (
            -beta * (2.0 - beta) * alpha**2 / (24.0 * torch.clamp((torch.clamp(shifted_forward * shifted_strike, min=EPS)) ** (1.0 - beta), min=EPS))
            + (rho * beta * nu * alpha) / (4.0 * torch.clamp((torch.clamp(shifted_forward * shifted_strike, min=EPS)) ** (0.5 * (1.0 - beta)), min=EPS))
            + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
        )
        ratio = torch.where(torch.abs(x_zeta) < ATM_TOL, torch.ones_like(diff), diff / x_zeta)
        return torch.clamp(alpha * ratio * (1.0 + correction * expiry), min=EPS)

    def implied_volatility(self, K, T, r=0.0, *, q=0.0, model: str = "lognormal", shift=None):
        expiry = self.tensor(T)
        forward = self._base_forward_from_state(self.F0, expiry, r, q)
        model_name = model.lower()
        if model_name == "lognormal":
            return self._hagan_lognormal_implied_volatility(K, expiry, forward=forward, shift=shift)
        if model_name == "normal":
            return self._hagan_normal_implied_volatility(K, expiry, forward=forward, shift=shift)
        raise ValueError("model must be 'lognormal' or 'normal'.")

    def simulate(self, S0, T, N, steps=100):
        N = int(N)
        steps = max(1, int(steps))
        expiry = self.tensor(T)
        dt = expiry / steps
        sqrt_dt = torch.sqrt(dt)

        beta = self.constrained_param("beta")
        rho = self.constrained_param("rho")
        nu = self.constrained_param("nu")

        forward = torch.clamp(self._base_forward_from_state(S0, expiry), min=EPS).expand(N).clone()
        alpha_t = torch.ones_like(forward) * self.constrained_param("alpha")
        rho_residual = torch.sqrt(torch.clamp(1.0 - rho**2, min=EPS))

        for _ in range(steps):
            z1 = torch.randn(N, device=self.device, dtype=forward.dtype)
            z2 = torch.randn(N, device=self.device, dtype=forward.dtype)
            z_alpha = rho * z1 + rho_residual * z2

            forward_base = torch.clamp(forward, min=EPS)
            diffusion = alpha_t * (forward_base**beta) * sqrt_dt * z1
            forward = torch.clamp(forward + diffusion, min=EPS)
            alpha_t = torch.clamp(alpha_t * torch.exp(-0.5 * nu**2 * dt + nu * sqrt_dt * z_alpha), min=EPS)

        return forward

    def option_price(self, K, T, r, option_type="call", N=10000, steps=100, *, q=0.0, model: str = "lognormal", shift=None):
        del N, steps
        option_style = option_type.lower()
        if option_style not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")

        expiry = self.tensor(T)
        rate = self.tensor(r)
        strike = self.tensor(K)
        forward = self._base_forward_from_state(self.F0, expiry, rate, q)
        model_name = model.lower()

        if model_name == "lognormal":
            shifted_forward, shifted_strike, _ = self._shifted_forward_and_strike(forward, strike, shift, positive_required=True)
            implied_volatility = self._hagan_lognormal_implied_volatility(strike, expiry, forward=forward, shift=shift)
            price = _black76_price(option_style, shifted_forward, shifted_strike, expiry, implied_volatility, rate)
        elif model_name == "normal":
            shifted_forward, shifted_strike, _ = self._shifted_forward_and_strike(forward, strike, shift, positive_required=False)
            implied_volatility = self._hagan_normal_implied_volatility(strike, expiry, forward=forward, shift=shift)
            price = _bachelier_price(option_style, shifted_forward, shifted_strike, expiry, implied_volatility, rate)
        else:
            raise ValueError("model must be 'lognormal' or 'normal'.")

        return float(price.item())
