Model calibration
=================

The calibration module provides two public entrypoints:

- ``CalibrationConfig`` for explicit configuration
- ``ModelCalibrator`` for calibration workflows and progress reporting

.. automodule:: torchquantlib.calibration.model_calibrator
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   import numpy as np
   import torch
   from torchquantlib import Heston, ModelCalibrator

   np.random.seed(42)
   torch.manual_seed(42)

   observed = Heston().simulate(S0=100.0, T=1.0, N=1000)

   calibrator = ModelCalibrator(
       model=Heston(),
       observed_data=observed.detach().cpu().numpy(),
       S0=100.0,
       T=1.0,
       lr=0.01,
   )
   calibrator.calibrate(num_epochs=100, steps=100, verbose=False)
   print(calibrator.get_calibrated_params())
