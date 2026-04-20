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
        with torch.no_grad():
            feller_gap = 2 * self.params["kappa"] * self.params["theta"] - self.params["sigma"] ** 2
            if feller_gap.item() <= 0:
                self.params["theta"].copy_(self.params["sigma"] ** 2 / (2 * torch.clamp(self.params["kappa"], min=1e-12)) + 1e-6)

    def simulate(self, S0, T, N, steps=100):
        del S0, steps
        horizon = self.tensor(T)
        kappa = torch.clamp(self.params["kappa"], min=0)
        theta = torch.clamp(self.params["theta"], min=0)
        sigma = torch.clamp(self.params["sigma"], min=0)
        r0 = torch.clamp(self.params["r0"], min=0)

        N = int(N)
        mean_reversion = torch.exp(-kappa * horizon)
        mean = theta + (r0 - theta) * mean_reversion

        if (kappa * horizon).item() >= 1.0:
            mean = theta

        if sigma.item() == 0:
            return mean.expand(N)

        stationary_variance = sigma**2 * theta / torch.clamp(2 * kappa, min=1e-12)
        variance = stationary_variance * (1 - torch.exp(-2 * kappa * horizon))
        noise = torch.randn(N, device=self.device, dtype=mean.dtype)
        return torch.relu(mean + torch.sqrt(torch.clamp(variance, min=0)) * noise)
