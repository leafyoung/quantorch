import torch

from torchquantlib import calculate_es, calculate_var


returns = torch.tensor([0.01, -0.02, 0.03, -0.01, 0.04, -0.03])
confidence_level = 0.95

var = calculate_var(returns, confidence_level)
es = calculate_es(returns, confidence_level)

print(f"VaR: {var.item()}")
print(f"Expected Shortfall: {es.item()}")
