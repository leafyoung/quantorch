import torch

from torchquantlib.models.stochastic_model import StochasticModel


class LMM(StochasticModel):
    """Libor Market Model."""

    parameter_constraints = {
        "volatilities": {"min": 1e-6},
    }

    def __init__(self, forward_rates_init, volatilities_init, correlations_init):
        if len(forward_rates_init) != len(volatilities_init):
            raise ValueError("forward_rates_init and volatilities_init must have the same length.")

        super().__init__(
            self.build_parameters(
                {
                    "forward_rates": forward_rates_init,
                    "volatilities": volatilities_init,
                }
            )
        )

        self.forward_rates = self.params["forward_rates"]
        self.volatilities = self.params["volatilities"]
        self.num_rates = int(self.forward_rates.numel())
        self.register_auxiliary_tensor("correlation_matrix", correlations_init)

        if self.correlation_matrix.shape != (self.num_rates, self.num_rates):
            raise ValueError("correlations_init must be a square matrix matching the number of rates.")

    def simulate(self, S0, T, N, steps=100):
        del S0, steps
        horizon = self.tensor(T)
        N = int(N)
        num_rates = self.num_rates

        forward_rates = self.forward_rates
        volatilities = torch.clamp(self.volatilities, min=0)
        corr_matrix = self.correlation_matrix.to(dtype=forward_rates.dtype)
        chol_matrix = torch.linalg.cholesky(corr_matrix)

        z = torch.randn(N, num_rates, device=self.device, dtype=forward_rates.dtype)
        correlated_noise = z @ chol_matrix.T
        sqrt_horizon = torch.sqrt(torch.clamp(horizon, min=0))

        if torch.all(forward_rates >= 0):
            base_scale = torch.mean(forward_rates)
            additive_terminal = forward_rates.unsqueeze(0) + base_scale * volatilities.unsqueeze(0) * sqrt_horizon * correlated_noise
            return torch.clamp(additive_terminal, min=0)

        additive_terminal = forward_rates.unsqueeze(0) + volatilities.unsqueeze(0) * sqrt_horizon * correlated_noise
        return additive_terminal
