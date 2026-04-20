import torch
from torch import Tensor


def calculate_binomial_tree_params(expiry: Tensor, volatility: Tensor, rate: Tensor, steps: int):
    dt = expiry / steps
    if dt.item() <= 0:
        return dt, torch.tensor(1.0, device=expiry.device, dtype=expiry.dtype), torch.tensor(1.0, device=expiry.device, dtype=expiry.dtype), torch.tensor(1.0, device=expiry.device, dtype=expiry.dtype)
    u = torch.exp(volatility * torch.sqrt(dt))
    d = 1 / u
    if torch.isclose(u, d):
        p = torch.tensor(1.0, device=expiry.device, dtype=expiry.dtype)
    else:
        p = (torch.exp(rate * dt) - d) / (u - d)
    return dt, u, d, p


def option_payoff(option_type: str, terminal_prices: Tensor, strike: Tensor) -> Tensor:
    zeros = torch.zeros_like(terminal_prices)
    if option_type == "call":
        return torch.maximum(terminal_prices - strike, zeros)
    if option_type == "put":
        return torch.maximum(strike - terminal_prices, zeros)
    raise ValueError("Invalid option type. Use 'call' or 'put'.")


def build_price_tree(spot: Tensor, u: Tensor, d: Tensor, steps: int) -> Tensor:
    price_tree = torch.zeros((steps + 1, steps + 1), device=spot.device, dtype=spot.dtype)
    for step in range(steps + 1):
        downs = torch.arange(step + 1, device=spot.device, dtype=spot.dtype)
        ups = step - downs
        price_tree[: step + 1, step] = spot * (u ** ups) * (d ** downs)
    return price_tree


def price_binomial_tree(
    option_type: str,
    option_style: str,
    spot: Tensor,
    strike: Tensor,
    expiry: Tensor,
    volatility: Tensor,
    rate: Tensor,
    steps: int,
    exercise_dates: Tensor | None = None,
) -> Tensor:
    if option_type not in {"call", "put"}:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    if option_style not in {"european", "american", "bermudan"}:
        raise ValueError("Invalid option style. Use 'european', 'american', or 'bermudan'.")
    if int(steps) <= 0:
        raise ValueError("steps must be a positive integer.")
    if expiry.item() <= 0:
        return option_payoff(option_type, spot, strike)
    if volatility.item() < 0:
        raise RuntimeError("volatility must be non-negative.")
    if torch.isclose(volatility, torch.zeros_like(volatility)):
        deterministic_terminal = spot * torch.exp(rate * expiry)
        return torch.exp(-rate * expiry) * option_payoff(option_type, deterministic_terminal, strike)

    steps = int(steps)
    dt, u, d, p = calculate_binomial_tree_params(expiry, volatility, rate, steps)
    discount = torch.exp(-rate * dt)
    price_tree = build_price_tree(spot, u, d, steps)
    value_tree = torch.zeros_like(price_tree)
    value_tree[: steps + 1, steps] = option_payoff(option_type, price_tree[: steps + 1, steps], strike)

    if option_style == "american":
        exercise_steps = set(range(steps))
    elif option_style == "bermudan" and exercise_dates is not None:
        exercise_steps = {int(step) for step in exercise_dates.flatten().tolist() if 0 <= int(step) < steps}
    else:
        exercise_steps = set()

    for step in range(steps - 1, -1, -1):
        continuation = (p * value_tree[: step + 1, step + 1] + (1 - p) * value_tree[1 : step + 2, step + 1]) * discount
        if step in exercise_steps:
            exercise = option_payoff(option_type, price_tree[: step + 1, step], strike)
            value_tree[: step + 1, step] = torch.maximum(exercise, continuation)
        else:
            value_tree[: step + 1, step] = continuation

    return value_tree[0, 0]
