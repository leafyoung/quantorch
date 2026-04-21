import torch

from torchquantlib.models.stochastic_model import StochasticModel


class HullWhiteModel(StochasticModel):
    """Hull-White short-rate model."""

    def zero_coupon_bond_price(self, maturity, r_t=None):
        maturity_tensor = self.tensor(maturity)
        if maturity_tensor.item() <= 0:
            return torch.ones((), device=self.device, dtype=self.params["r0"].dtype)

        a = torch.clamp(self.params["a"], min=1e-12)
        sigma = torch.clamp(self.params["sigma"], min=0.0)
        short_rate = self.params["r0"] if r_t is None else self.tensor(r_t)

        b_term = (1 - torch.exp(-a * maturity_tensor)) / a
        a_term = torch.exp((sigma**2 / (4 * a**3)) * (2 * a * maturity_tensor + 4 * torch.exp(-a * maturity_tensor) - torch.exp(-2 * a * maturity_tensor) - 3))
        return a_term * torch.exp(-b_term * short_rate)

    def zero_coupon_yield(self, maturity, r_t=None):
        maturity_tensor = self.tensor(maturity)
        price = self.zero_coupon_bond_price(maturity_tensor, r_t=r_t)
        return -torch.log(torch.clamp(price, min=1e-12)) / torch.clamp(maturity_tensor, min=1e-12)

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
        r0 = self.params["r0"]

        N = int(N)
        mean = r0 * torch.exp(-a * horizon)

        if sigma.item() == 0:
            return mean.expand(N)

        variance = sigma**2 * (1 - torch.exp(-2 * a * horizon)) / torch.clamp(2 * a, min=1e-12)
        noise = torch.randn(N, device=self.device, dtype=mean.dtype)
        return mean + torch.sqrt(torch.clamp(variance, min=0)) * noise
