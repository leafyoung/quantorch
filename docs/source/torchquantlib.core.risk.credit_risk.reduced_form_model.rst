Reduced Form Model
==================

.. currentmodule:: torchquantlib.core.risk.credit_risk.reduced_form_model

TorchQuant exposes a simple reduced-form expected-loss calculation through
``reduced_form_model``.

.. autofunction:: reduced_form_model

Example
-------

.. code-block:: python

   import torch
   from torchquantlib import reduced_form_model

   expected_loss = reduced_form_model(
       lambda_0=torch.tensor(0.02),
       default_intensity=torch.tensor(0.05),
       recovery_rate=torch.tensor(0.4),
       time=torch.tensor(1.0),
   )
   print(expected_loss)
