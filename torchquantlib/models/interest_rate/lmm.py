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
        dt = self.tensor(T / steps)
        N = int(N)
        steps = int(steps)
        num_rates = self.num_rates

        forward_rates = self.forward_rates
        volatilities = self.constrained_param("volatilities")
        corr_matrix = self.correlation_matrix
        chol_matrix = torch.linalg.cholesky(corr_matrix)

        F = torch.zeros(N, num_rates, steps, device=self.device)
        F[:, :, 0] = forward_rates

        sqrt_dt = torch.sqrt(dt)
        dtau = dt

        for t in range(1, steps):
            Z = torch.randn(N, num_rates, device=self.device)
            W = Z @ chol_matrix.T

            for i in range(num_rates):
                F_t_minus = F[:, i, t - 1]
                sigma_i = volatilities[i]
                drift = torch.zeros(N, device=self.device)

                for j in range(i + 1, num_rates):
                    sigma_j = volatilities[j]
                    corr_ij = corr_matrix[i, j]
                    F_j = F[:, j, t - 1]
                    drift += dtau * sigma_i * sigma_j * corr_ij * F_j / (1 + dtau * F_j)

                dF = drift * dt - sigma_i * F_t_minus * W[:, i] * sqrt_dt
                F[:, i, t] = F_t_minus + dF

        return F[:, :, -1]
