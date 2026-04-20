import torch

from torchquantlib.models.stochastic_model import StochasticModel


class Heston(StochasticModel):
    """Heston stochastic volatility model."""

    parameter_constraints = {
        "theta": {"min": 1e-6},
        "sigma_v": {"min": 1e-6},
        "rho": {"min": -0.999, "max": 0.999},
        "v0": {"min": 1e-6},
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

    def simulate(self, S0, T, N, steps=100):
        dt = self.tensor(T / steps)
        mu = self.params["mu"]
        kappa = self.params["kappa"]
        theta = self.constrained_param("theta")
        sigma_v = self.constrained_param("sigma_v")
        rho = self.constrained_param("rho")
        v0 = self.constrained_param("v0")

        S0 = self.tensor(S0)
        N = int(N)
        steps = int(steps)

        S = torch.zeros(N, steps, device=self.device)
        v = torch.zeros(N, steps, device=self.device)
        S[:, 0] = S0
        v[:, 0] = v0

        sqrt_dt = torch.sqrt(dt)
        correlation_scale = torch.sqrt(1 - rho**2)

        for t in range(1, steps):
            z1 = torch.randn(N, device=self.device)
            z2 = torch.randn(N, device=self.device)
            w_s = z1 * sqrt_dt
            w_v = (rho * z1 + correlation_scale * z2) * sqrt_dt

            v_t_minus = torch.relu(v[:, t - 1])
            dv = kappa * (theta - v_t_minus) * dt + sigma_v * torch.sqrt(v_t_minus) * w_v
            v[:, t] = v_t_minus + dv

            dS = S[:, t - 1] * (mu * dt + torch.sqrt(v_t_minus) * w_s)
            S[:, t] = S[:, t - 1] + dS

        return S[:, -1]

    def option_price(self, S0, K, T, r, option_type="call", N=10000, steps=100):
        terminal_prices = self.simulate(S0, T, N, steps)
        option_style = option_type.lower()

        if option_style == "call":
            payoff = torch.relu(terminal_prices - K)
        elif option_style == "put":
            payoff = torch.relu(K - terminal_prices)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        discount_factor = torch.exp(-self.tensor(r * T))
        price = discount_factor * torch.mean(payoff)
        return price.item()
