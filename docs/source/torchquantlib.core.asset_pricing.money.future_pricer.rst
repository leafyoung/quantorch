Future Pricer
=============

.. automodule:: torchquantlib.core.asset_pricing.money.future_pricer
   :members:
   :undoc-members:
   :show-inheritance:

``future_pricer`` prices a futures contract using spot, domestic rate, foreign rate,
and expiry.

.. autofunction:: torchquantlib.core.asset_pricing.money.future_pricer.future_pricer

Example
-------

.. code-block:: python

   import torch
   from torchquantlib import future_pricer

   spot = torch.tensor(100.0)
   domestic_rate = torch.tensor(0.05)
   foreign_rate = torch.tensor(0.02)
   expiry = torch.tensor(1.0)

   futures_price = future_pricer(spot, domestic_rate, foreign_rate, expiry)
   print(f"Futures Price: {futures_price.item():.2f}")
