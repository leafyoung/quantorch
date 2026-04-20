import torch

from torchquantlib.models.stochastic_model import StochasticModel


class SABR(StochasticModel):
    """SABR stochastic volatility model."""

    parameter_constraints = {
        "alpha": {"min": 1e-6},
        "beta": {"min": 0.0, "max": 1.0},
        "rho": {"min": -0.999, "max": 0.999},
        "nu": {"min": 1e-6},
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

    def simulate(self, S0, T, N, steps=100):
        dt = self.tensor(T / steps)
        alpha = self.constrained_param("alpha")
        beta = self.constrained_param("beta")
        rho = self.constrained_param("rho")
        nu = self.constrained_param("nu")

        N = int(N)
        steps = int(steps)

        F = torch.zeros(N, steps, device=self.device)
        alpha_t = torch.zeros(N, steps, device=self.device)
        F[:, 0] = self.F0
        alpha_t[:, 0] = alpha

        sqrt_dt = torch.sqrt(dt)
        correlation_scale = torch.sqrt(1 - rho**2)

        for t in range(1, steps):
            z1 = torch.randn(N, device=self.device)
            z2 = torch.randn(N, device=self.device)
            w_f = z1 * sqrt_dt
            w_alpha = (rho * z1 + correlation_scale * z2) * sqrt_dt

            F_t_minus = F[:, t - 1]
            alpha_t_minus = alpha_t[:, t - 1]

            dF = alpha_t_minus * (F_t_minus**beta) * w_f
            F[:, t] = F_t_minus + dF

            dalpha = nu * alpha_t_minus * w_alpha
            alpha_t[:, t] = alpha_t_minus + dalpha

        return F[:, -1]

    def option_price(self, K, T, r, option_type="call", N=10000, steps=100):
        terminal_forward_rates = self.simulate(S0=self.F0, T=T, N=N, steps=steps)
        option_style = option_type.lower()

        if option_style == "call":
            payoff = torch.relu(terminal_forward_rates - K)
        elif option_style == "put":
            payoff = torch.relu(K - terminal_forward_rates)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        discount_factor = torch.exp(-self.tensor(r * T))
        price = discount_factor * torch.mean(payoff)
        return price.item()
