from torch import Tensor


def scenario_analysis(portfolio_value: Tensor, scenarios: Tensor) -> Tensor:
    """Apply scenario multipliers to a portfolio value."""
    if not isinstance(portfolio_value, Tensor):
        raise TypeError("portfolio_value must be a torch.Tensor.")
    if not isinstance(scenarios, Tensor):
        raise TypeError("scenarios must be a torch.Tensor.")
    return portfolio_value * scenarios
