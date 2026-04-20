import torch

from torchquantlib.models.stochastic_model import StochasticModel


class DupireLocalVol(StochasticModel):
    """Dupire local volatility model."""

    def __init__(self, local_vol_func):
        self.local_vol_func = local_vol_func
        super().__init__(params={})

    def _evaluate_local_volatility(self, spot: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        try:
            sigma = self.local_vol_func(spot, time, self.device)
        except TypeError:
            sigma = self.local_vol_func(spot, time)
        sigma_tensor = sigma if isinstance(sigma, torch.Tensor) else torch.full_like(spot, float(sigma))
        if torch.any(sigma_tensor < 0):
            raise RuntimeError("Negative volatility encountered")
        return sigma_tensor.to(device=self.device, dtype=spot.dtype)

    def simulate(self, S0, T, N, steps=100, rate=0.0):
        if not isinstance(S0, (int, float)):
            raise TypeError("S0 must be a number.")
        if not isinstance(T, (int, float)):
            raise TypeError("T must be a number.")
        if not isinstance(N, int):
            raise TypeError("N must be an integer.")
        if not isinstance(rate, (int, float)):
            raise TypeError("rate must be a number.")

        steps = int(steps)
        dt = self.tensor(T / steps)
        spot = torch.full((N,), float(S0), device=self.device)
        rate_tensor = self.tensor(rate)
        sqrt_dt = torch.sqrt(dt)

        for step in range(steps):
            time = (step + 1) * dt
            sigma = self._evaluate_local_volatility(spot, time)
            z = torch.randn(N, device=self.device, dtype=spot.dtype)
            drift = rate_tensor * dt
            diffusion = sigma * sqrt_dt * z
            spot = spot * torch.exp(drift + diffusion)

        return spot
