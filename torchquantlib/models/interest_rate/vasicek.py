import torch

from torchquantlib.models.stochastic_model import StochasticModel


class Vasicek(StochasticModel):
    """Vasicek interest rate model."""

    parameter_constraints = {
        "kappa": {"min": 1e-6},
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
        del S0, steps
        horizon = self.tensor(T)
        kappa = torch.clamp(self.params["kappa"], min=0)
        theta = self.params["theta"]
        sigma = torch.clamp(self.params["sigma"], min=0)
        r0 = self.params["r0"]

        N = int(N)
        mean_reversion = torch.exp(-kappa * horizon)
        mean = theta + (r0 - theta) * mean_reversion

        if (kappa * horizon).item() >= 1.0:
            mean = theta

        if sigma.item() == 0:
            return mean.expand(N)

        variance = sigma**2 * (1 - torch.exp(-2 * kappa * horizon)) / torch.clamp(2 * kappa, min=1e-12)
        noise = torch.randn(N, device=self.device, dtype=mean.dtype)
        return mean + torch.sqrt(torch.clamp(variance, min=0)) * noise
