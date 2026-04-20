import torch

from torchquantlib.models.stochastic_model import StochasticModel


class HullWhiteModel(StochasticModel):
    """Hull-White short-rate model."""

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
        dt = self.tensor(T / steps)
        a = self.constrained_param("a")
        sigma = self.constrained_param("sigma")
        r0 = self.params["r0"]

        N = int(N)
        steps = int(steps)

        r = torch.zeros(N, steps, device=self.device)
        r[:, 0] = r0

        for t in range(1, steps):
            r_t_minus = r[:, t - 1]
            dr = -a * r_t_minus * dt + sigma * torch.sqrt(dt) * torch.randn(N, device=self.device)
            r[:, t] = r_t_minus + dr

        return r[:, -1]
