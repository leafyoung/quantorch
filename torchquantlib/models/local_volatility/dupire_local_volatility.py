import torch

from torchquantlib.models.stochastic_model import StochasticModel


class DupireLocalVol(StochasticModel):
    """Dupire local volatility model."""

    def __init__(self, local_vol_func):
        self.local_vol_func = local_vol_func
        super().__init__(params={})

    def simulate(self, S0, T, N, rate, steps=100):
        if not isinstance(S0, (int, float)):
            raise TypeError("S0 must be a number.")
        if not isinstance(T, (int, float)):
            raise TypeError("T must be a number.")
        if not isinstance(N, int):
            raise TypeError("N must be an integer.")
        if not isinstance(rate, (int, float)):
            raise TypeError("rate must be a number.")

        dt = self.tensor(T / steps)
        S0_tensor = self.tensor(S0)
        rate_tensor = self.tensor(rate)
        steps = int(steps)

        S = torch.zeros(N, steps, device=self.device)
        S[:, 0] = S0_tensor

        for t in range(1, steps):
            t_curr = t * dt
            S_t_minus = S[:, t - 1]
            sigma_t = self.local_vol_func(S_t_minus, t_curr, self.device)
            Z = torch.randn(N, device=self.device)
            dS = rate_tensor * S_t_minus * dt + sigma_t * S_t_minus * torch.sqrt(dt) * Z
            S[:, t] = S_t_minus + dS

        return S[:, -1]
