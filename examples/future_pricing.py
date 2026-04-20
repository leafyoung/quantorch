import torch

from torchquantlib import future_pricer


domestic_rate = torch.tensor(0.05)
foreign_rate = torch.tensor(0.03)
spot = torch.tensor(1.0)
expiry = torch.tensor(1.0)

price = future_pricer(spot, domestic_rate, foreign_rate, expiry)
print(f"Currency Price: {price.item()}")
