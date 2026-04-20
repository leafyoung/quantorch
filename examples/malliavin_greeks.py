import torch

from torchquantlib import MalliavinGreeks


greeks = MalliavinGreeks()
results = greeks.european_option_greeks(
    S0=torch.tensor(100.0),
    K=torch.tensor(100.0),
    T=torch.tensor(1.0),
    r=torch.tensor(0.05),
    sigma=torch.tensor(0.2),
)
print(results)
