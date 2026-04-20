import torch

from torchquantlib import DupireLocalVol


def local_vol_func(spot, time, device):
    del time, device
    return torch.full_like(spot, 0.2)


model = DupireLocalVol(local_vol_func)
terminal_prices = model.simulate(S0=100.0, T=1.0, N=1000, rate=0.05, steps=252)
print(f"Terminal Prices: {terminal_prices[:5]}")
