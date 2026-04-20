import torch

from torchquantlib.models.stochastic_model import StochasticModel


class BlackKarasinski(StochasticModel):
    """Black-Karasinski short-rate model."""

    parameter_constraints = {
        "a": {"min": 1e-6},
        "sigma": {"min": 1e-6},
    }

    def __init__(self, a_init=0.1, sigma_init=0.01, r0_init=0.03):
        super().__init__(
            self.build_parameters(
                {
                    "a": a_init,
                    "sigma": sigma_init,
                    "r0": r0_init,
                }
            )
        )

    def simulate(self, S0, T, N, steps=100):
        del S0, steps
        horizon = self.tensor(T)
        a = torch.clamp(self.params["a"], min=0)
        sigma = torch.clamp(self.params["sigma"], min=0)
        r0 = torch.clamp(self.params["r0"], min=1e-8)

        N = int(N)
        log_r0 = torch.log(r0)
        mean = log_r0 * torch.exp(-a * horizon)

        if sigma.item() == 0:
            return torch.exp(mean).expand(N)

        variance = sigma**2 * (1 - torch.exp(-2 * a * horizon)) / torch.clamp(2 * a, min=1e-12)
        noise = torch.randn(N, device=self.device, dtype=mean.dtype)
        return torch.exp(mean + torch.sqrt(torch.clamp(variance, min=0)) * noise)
