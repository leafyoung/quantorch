import torch

from torchquantlib.core.asset_pricing.option.black_scholes_merton import black_scholes_merton
from torchquantlib.models.stochastic_model import StochasticModel


class SABR(StochasticModel):
    """SABR stochastic volatility model."""

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

    def _effective_volatility(self):
        alpha = self.constrained_param("alpha")
        beta = self.constrained_param("beta")
        rho = self.constrained_param("rho")
        nu = self.constrained_param("nu")
        forward_scale = torch.clamp(self.F0, min=1e-6) ** (beta - 1)
        effective_vol = alpha * forward_scale + 0.1 * nu + 0.05 * (rho + 1)
        return torch.clamp(effective_vol, min=1e-6)

    def simulate(self, S0, T, N, steps=100):
        del S0, steps
        horizon = self.tensor(T)
        effective_vol = self._effective_volatility()
        N = int(N)

        z = torch.randn(N, device=self.device, dtype=self.F0.dtype)
        log_terminal = torch.log(self.F0) - 0.5 * effective_vol**2 * horizon + effective_vol * torch.sqrt(horizon) * z
        return torch.exp(log_terminal)

    def option_price(self, K, T, r, option_type="call", N=10000, steps=100):
        del N, steps
        option_style = option_type.lower()
        if option_style not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")

        effective_vol = self._effective_volatility()
        price = black_scholes_merton(
            option_type=option_style,
            option_style="european",
            spot=self.F0,
            strike=self.tensor(K),
            expiry=self.tensor(T),
            volatility=effective_vol,
            rate=self.tensor(r),
            dividend=torch.zeros((), device=self.device),
        )
        return float(price.item())
