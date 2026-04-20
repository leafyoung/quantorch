Examples
========

This section provides short examples built on the supported public API.

Model calibration
-----------------

.. code-block:: python

   import numpy as np
   import torch
   from torchquantlib import Heston, ModelCalibrator

   N_observed = 1000
   S0 = 100.0
   T = 1.0

   np.random.seed(42)
   torch.manual_seed(42)

   heston_true = Heston(kappa_init=2.0, theta_init=0.04, sigma_v_init=0.3, rho_init=-0.7, v0_init=0.04, mu_init=0.05)
   observed = heston_true.simulate(S0=S0, T=T, N=N_observed)

   calibrator = ModelCalibrator(
       model=Heston(),
       observed_data=observed.detach().cpu().numpy(),
       S0=S0,
       T=T,
       lr=0.01,
   )
   calibrator.calibrate(num_epochs=100, steps=100, verbose=False)
   print(calibrator.get_calibrated_params())

Option pricing
--------------

.. code-block:: python

   import torch
   from torchquantlib import american_option, asian_option, bermudan_option

   spot = torch.tensor(100.0)
   strike = torch.tensor(105.0)
   expiry = torch.tensor(1.0)
   volatility = torch.tensor(0.2)
   rate = torch.tensor(0.05)

   american_price = american_option("call", spot, strike, expiry, volatility, rate, 100)
   asian_price = asian_option("call", spot, strike, expiry, volatility, rate, 100)
   bermudan_price = bermudan_option(
       "call",
       spot,
       strike,
       expiry,
       volatility,
       rate,
       100,
       torch.tensor([25, 50, 75]),
   )

   print(american_price.item(), asian_price.item(), bermudan_price.item())
