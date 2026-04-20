import torch

from torchquantlib import Vasicek


model = Vasicek(
    kappa_init=0.5,
    theta_init=0.04,
    sigma_init=0.01,
    r0_init=0.03,
)
rates = model.simulate(S0=0.0, T=1.0, N=1000, steps=252)
print(f"Interest Rates: {rates}")
