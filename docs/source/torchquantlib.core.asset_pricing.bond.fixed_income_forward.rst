Fixed Income Forward
====================

.. automodule:: torchquantlib.core.asset_pricing.bond.fixed_income_forward
   :members:
   :undoc-members:
   :show-inheritance:

Use ``fixed_income_forward`` to price a simple fixed-income forward contract.

.. autofunction:: torchquantlib.core.asset_pricing.bond.fixed_income_forward.fixed_income_forward

Example
-------

.. code-block:: python

   import torch
   from torchquantlib import fixed_income_forward

   price = fixed_income_forward(
       torch.tensor(1000.0),
       torch.tensor(0.05),
       torch.tensor(2.0),
       torch.tensor(0.06),
   )
   print(price)
