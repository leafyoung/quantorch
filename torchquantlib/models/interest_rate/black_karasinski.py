import torch

from torchquantlib.models.stochastic_model import StochasticModel


class BlackKarasinski(StochasticModel):
    """Black-Karasinski short-rate model."""

    def zero_coupon_bond_price(self, maturity, r_t=None, integration_steps: int = 256):
        maturity_tensor = self.tensor(maturity)
        if maturity_tensor.item() <= 0:
            return torch.ones((), device=self.device, dtype=self.params["r0"].dtype)

        a = torch.clamp(self.params["a"], min=1e-12)
        sigma = torch.clamp(self.params["sigma"], min=0.0)
        short_rate = torch.clamp(self.params["r0"] if r_t is None else self.tensor(r_t), min=1e-8)

        times = torch.linspace(0.0, maturity_tensor.item(), max(2, int(integration_steps)), device=self.device, dtype=short_rate.dtype)
        mean_log_r = torch.log(short_rate) * torch.exp(-a * times)
        variance_log_r = sigma**2 * (1 - torch.exp(-2 * a * times)) / torch.clamp(2 * a, min=1e-12)
        expected_short_rate = torch.exp(mean_log_r + 0.5 * variance_log_r)
        integral = torch.trapz(expected_short_rate, times)
        return torch.exp(-integral)

    def zero_coupon_yield(self, maturity, r_t=None, integration_steps: int = 256):
        maturity_tensor = self.tensor(maturity)
        price = self.zero_coupon_bond_price(maturity_tensor, r_t=r_t, integration_steps=integration_steps)
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
        r0 = torch.clamp(self.params["r0"], min=1e-8)

        N = int(N)
        log_r0 = torch.log(r0)
        mean = log_r0 * torch.exp(-a * horizon)

        if sigma.item() == 0:
            return torch.exp(mean).expand(N)

        variance = sigma**2 * (1 - torch.exp(-2 * a * horizon)) / torch.clamp(2 * a, min=1e-12)
        noise = torch.randn(N, device=self.device, dtype=mean.dtype)
        return torch.exp(mean + torch.sqrt(torch.clamp(variance, min=0)) * noise)
