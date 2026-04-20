import torch

from torchquantlib.models.stochastic_model import StochasticModel


class CIR(StochasticModel):
    """Cox-Ingersoll-Ross interest rate model."""

    parameter_constraints = {
        "kappa": {"min": 1e-6},
        "theta": {"min": 1e-6},
        "sigma": {"min": 1e-6},
    }

    def __init__(self, kappa_init=0.1, theta_init=0.05, sigma_init=0.01, r0_init=0.03):
        super().__init__(
            self.build_parameters(
                {
                    "kappa": kappa_init,
                    "theta": theta_init,
                    "sigma": sigma_init,
                    "r0": r0_init,
                }
            )
        )

    def simulate(self, S0, T, N, steps=100):
        dt = self.tensor(T / steps)
        kappa = self.constrained_param("kappa")
        theta = self.constrained_param("theta")
        sigma = self.constrained_param("sigma")
        r0 = self.params["r0"]

        N = int(N)
        steps = int(steps)

        r = torch.zeros(N, steps, device=self.device)
        r[:, 0] = r0

        for t in range(1, steps):
            r_t_minus = torch.relu(r[:, t - 1])
            dr = kappa * (theta - r_t_minus) * dt + sigma * torch.sqrt(r_t_minus * dt) * torch.randn(N, device=self.device)
            r[:, t] = r_t_minus + dr

        return r[:, -1]
