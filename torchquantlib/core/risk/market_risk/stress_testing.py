from torch import Tensor


def stress_test(portfolio_value: Tensor, stress_scenarios: Tensor) -> Tensor:
    """Apply percentage stress scenarios to a portfolio value."""
    if not isinstance(portfolio_value, Tensor):
        raise TypeError("portfolio_value must be a torch.Tensor.")
    if not isinstance(stress_scenarios, Tensor):
        raise TypeError("stress_scenarios must be a torch.Tensor.")
    return portfolio_value * (1 + stress_scenarios)
