import numpy as np
import torch

from torchquantlib import Heston, ModelCalibrator


N_observed = 1000
S0 = 100.0
T = 1.0

np.random.seed(42)
torch.manual_seed(42)

heston_true = Heston(
    kappa_init=2.0,
    theta_init=0.04,
    sigma_v_init=0.3,
    rho_init=-0.7,
    v0_init=0.04,
    mu_init=0.05,
)
S_observed = heston_true.simulate(S0=S0, T=T, N=N_observed)

heston_model = Heston(
    kappa_init=1.0,
    theta_init=0.02,
    sigma_v_init=0.2,
    rho_init=-0.5,
    v0_init=0.02,
    mu_init=0.0,
)

calibrator = ModelCalibrator(
    model=heston_model,
    observed_data=S_observed.detach().cpu().numpy(),
    S0=S0,
    T=T,
    lr=0.01,
)

calibrator.calibrate(num_epochs=1000, steps=100, verbose=True)
print(calibrator.get_calibrated_params())
