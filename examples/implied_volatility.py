import torch

from torchquantlib import implied_volatility


market_price = torch.tensor(10.0)
spot = torch.tensor(100.0)
strike = torch.tensor(105.0)
expiry = torch.tensor(1.0)
rate = torch.tensor(0.05)

iv = implied_volatility(market_price, spot, strike, expiry, rate, is_call=True)
print(f"Implied Volatility: {iv.item()}")
