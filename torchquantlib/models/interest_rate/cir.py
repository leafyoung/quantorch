import torch

from torchquantlib.models.stochastic_model import StochasticModel


class CIR(StochasticModel):
    """Cox-Ingersoll-Ross interest rate model."""

    def zero_coupon_bond_price(self, maturity, r_t=None):
        maturity_tensor = self.tensor(maturity)
        if maturity_tensor.item() <= 0:
            return torch.ones((), device=self.device, dtype=self.params["r0"].dtype)

        kappa = torch.clamp(self.params["kappa"], min=1e-12)
        theta = torch.clamp(self.params["theta"], min=0.0)
        sigma = torch.clamp(self.params["sigma"], min=1e-12)
        short_rate = torch.clamp(self.params["r0"] if r_t is None else self.tensor(r_t), min=0.0)

        gamma = torch.sqrt(kappa**2 + 2 * sigma**2)
        exp_gamma_t = torch.exp(gamma * maturity_tensor)
        denominator = (gamma + kappa) * (exp_gamma_t - 1) + 2 * gamma
        b_term = 2 * (exp_gamma_t - 1) / torch.clamp(denominator, min=1e-12)
        a_term = torch.clamp(
            2 * gamma * torch.exp((kappa + gamma) * maturity_tensor / 2) / torch.clamp(denominator, min=1e-12),
            min=1e-12,
        ) ** (2 * kappa * theta / sigma**2)
        return a_term * torch.exp(-b_term * short_rate)

    def zero_coupon_yield(self, maturity, r_t=None):
        maturity_tensor = self.tensor(maturity)
        price = self.zero_coupon_bond_price(maturity_tensor, r_t=r_t)
        return -torch.log(torch.clamp(price, min=1e-12)) / torch.clamp(maturity_tensor, min=1e-12)

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
