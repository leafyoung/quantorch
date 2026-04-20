Structural Model
================

.. currentmodule:: torchquantlib.core.risk.credit_risk.structural_model

TorchQuant exposes the structural credit-risk model through ``merton_model``.

.. autofunction:: merton_model

Example
-------

.. code-block:: python

   import torch
   from torchquantlib import merton_model

   equity_value = merton_model(
       asset_value=torch.tensor(100.0),
       debt=torch.tensor(80.0),
       volatility=torch.tensor(0.2),
       rate=torch.tensor(0.05),
       expiry=torch.tensor(1.0),
   )
   print(equity_value)
