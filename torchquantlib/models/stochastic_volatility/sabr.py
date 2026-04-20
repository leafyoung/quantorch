import math

import torch

from torchquantlib.models.stochastic_model import StochasticModel

EPS = 1e-8
ATM_TOL = 1e-7


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


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


class SABR(StochasticModel):
    """SABR model with Hagan lognormal implied volatility pricing.

    ``option_price`` uses Hagan's asymptotic lognormal implied volatility
    approximation together with Black-76 pricing. ``simulate`` provides a
    simple Euler-style discretization of the SABR forward and volatility
    dynamics for compatibility with the stochastic-model API.
    """

    parameter_constraints = {
        "alpha": {"min": 1.0001e-6},
        "beta": {"min": 0.0, "max": 1.0},
        "rho": {"min": -0.9989, "max": 0.9989},
        "nu": {"min": 1.0001e-6},
    }

    def __init__(self, alpha_init=0.2, beta_init=0.5, rho_init=0.0, nu_init=0.3, F0=100.0):
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

    def _forward_from_state(self, base_level, expiry, rate=None):
        forward = self.tensor(base_level if base_level is not None else self.F0)
        forward = torch.clamp(forward, min=EPS)
        if rate is None:
            return forward
        return torch.clamp(forward * torch.exp(self.tensor(rate) * expiry), min=EPS)

    def _atm_implied_volatility(self, forward: torch.Tensor, expiry: torch.Tensor) -> torch.Tensor:
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

    def _hagan_lognormal_implied_volatility(
        self,
        strike,
        expiry,
        *,
        forward: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expiry = self.tensor(expiry)
        if expiry.item() <= 0:
            return torch.zeros((), device=self.device)

        alpha = self.constrained_param("alpha")
        beta = self.constrained_param("beta")
        rho = self.constrained_param("rho")
        nu = self.constrained_param("nu")

        forward = self._forward_from_state(self.F0 if forward is None else forward, expiry)
        strike = torch.clamp(self.tensor(strike), min=EPS)

        log_fk = torch.log(forward / strike)
        if torch.abs(log_fk).item() < ATM_TOL:
            return self._atm_implied_volatility(forward, expiry)

        one_minus_beta = 1.0 - beta
        fk_beta = torch.clamp((forward * strike) ** (0.5 * one_minus_beta), min=EPS)
        log_fk_sq = log_fk**2
        log_fk_4 = log_fk_sq**2

        z = (nu / torch.clamp(alpha, min=EPS)) * fk_beta * log_fk
        sqrt_term = torch.sqrt(torch.clamp(1.0 - 2.0 * rho * z + z**2, min=EPS))
        x_z = torch.log(torch.clamp((sqrt_term + z - rho) / (1.0 - rho), min=EPS))

        log_denominator = 1.0 + ((one_minus_beta**2) / 24.0) * log_fk_sq + ((one_minus_beta**4) / 1920.0) * log_fk_4
        prefactor = alpha / torch.clamp(fk_beta * log_denominator, min=EPS)

        correction = (
            ((one_minus_beta**2) / 24.0) * (alpha**2 / torch.clamp((forward * strike) ** one_minus_beta, min=EPS))
            + (rho * beta * nu * alpha) / (4.0 * torch.clamp(fk_beta, min=EPS))
            + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
        )

        ratio = torch.where(torch.abs(x_z) < ATM_TOL, torch.ones_like(z), z / x_z)
        implied_volatility = prefactor * ratio * (1.0 + correction * expiry)
        return torch.clamp(implied_volatility, min=EPS)

    def implied_volatility(self, K, T, r=0.0):
        expiry = self.tensor(T)
        forward = self._forward_from_state(self.F0, expiry, r)
        return self._hagan_lognormal_implied_volatility(K, expiry, forward=forward)

    def simulate(self, S0, T, N, steps=100):
        N = int(N)
        steps = max(1, int(steps))
        expiry = self.tensor(T)
        dt = expiry / steps
        sqrt_dt = torch.sqrt(dt)

        beta = self.constrained_param("beta")
        rho = self.constrained_param("rho")
        nu = self.constrained_param("nu")

        forward = self._forward_from_state(S0, expiry).expand(N).clone()
        alpha_t = torch.ones_like(forward) * self.constrained_param("alpha")

        rho_residual = torch.sqrt(torch.clamp(1.0 - rho**2, min=EPS))

        for _ in range(steps):
            z1 = torch.randn(N, device=self.device, dtype=forward.dtype)
            z2 = torch.randn(N, device=self.device, dtype=forward.dtype)
            z_alpha = rho * z1 + rho_residual * z2

            forward_base = torch.clamp(forward, min=EPS)
            diffusion = alpha_t * (forward_base**beta) * sqrt_dt * z1
            forward = torch.clamp(forward + diffusion, min=EPS)

            alpha_t = torch.clamp(
                alpha_t * torch.exp(-0.5 * nu**2 * dt + nu * sqrt_dt * z_alpha),
                min=EPS,
            )

        return forward

    def option_price(self, K, T, r, option_type="call", N=10000, steps=100):
        del N, steps
        option_style = option_type.lower()
        if option_style not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")

        expiry = self.tensor(T)
        rate = self.tensor(r)
        strike = torch.clamp(self.tensor(K), min=EPS)
        forward = self._forward_from_state(self.F0, expiry, rate)
        implied_volatility = self._hagan_lognormal_implied_volatility(strike, expiry, forward=forward)
        price = _black76_price(option_style, forward, strike, expiry, implied_volatility, rate)
        return float(price.item())
