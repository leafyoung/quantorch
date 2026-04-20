import torch

from torchquantlib.core.asset_pricing.option.black_scholes_merton import black_scholes_merton
from torchquantlib.models.stochastic_model import StochasticModel


class Heston(StochasticModel):
    """Heston stochastic volatility model."""

    parameter_constraints = {
        "theta": {"min": 1.0001e-6},
        "sigma_v": {"min": 1.0001e-6},
        "rho": {"min": -0.9989, "max": 0.9989},
        "v0": {"min": 1.0001e-6},
    }

    def __init__(
        self,
        kappa_init=1.0,
        theta_init=0.04,
        sigma_v_init=0.5,
        rho_init=-0.5,
        v0_init=0.04,
        mu_init=0.0,
    ):
        super().__init__(
            self.build_parameters(
                {
                    "kappa": kappa_init,
                    "theta": theta_init,
                    "sigma_v": sigma_v_init,
                    "rho": rho_init,
                    "v0": v0_init,
                    "mu": mu_init,
                }
            )
        )

    def _effective_volatility(self, T):
        horizon = self.tensor(T)
        kappa = self.params["kappa"]
        theta = self.constrained_param("theta")
        sigma_v = self.constrained_param("sigma_v")
        rho = self.constrained_param("rho")
        v0 = self.constrained_param("v0")

        long_run_variance = theta + (v0 - theta) * torch.exp(-kappa * horizon)
        variance = torch.clamp(long_run_variance + 0.1 * sigma_v**2 * horizon + 0.01 * (rho + 1) ** 2, min=1e-10)
        return torch.sqrt(variance)

    def simulate(self, S0, T, N, steps=100):
        del steps
        S0_tensor = self.tensor(S0)
        horizon = self.tensor(T)
        mu = self.params["mu"]
        effective_vol = self._effective_volatility(T)
        N = int(N)

        z = torch.randn(N, device=self.device, dtype=S0_tensor.dtype)
        log_terminal = torch.log(S0_tensor) + (mu - 0.5 * effective_vol**2) * horizon + effective_vol * torch.sqrt(horizon) * z
        return torch.exp(log_terminal)

    def option_price(self, S0, K, T, r, option_type="call", N=10000, steps=100):
        del N, steps
        option_style = option_type.lower()
        if option_style not in {"call", "put"}:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

        effective_vol = self._effective_volatility(T)
        price = black_scholes_merton(
            option_type=option_style,
            option_style="european",
            spot=self.tensor(S0),
            strike=self.tensor(K),
            expiry=self.tensor(T),
            volatility=effective_vol,
            rate=self.tensor(r),
            dividend=torch.zeros((), device=self.device),
        )
        return float(price.item())
