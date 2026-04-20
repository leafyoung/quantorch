Bond Pricer
===========

.. automodule:: torchquantlib.core.asset_pricing.bond.bond_pricer
   :members:
   :undoc-members:
   :show-inheritance:

The bond pricer module provides small helper functions for standard bond cash-flow calculations.

Zero Coupon Bond
----------------

.. autofunction:: torchquantlib.core.asset_pricing.bond.bond_pricer.zero_coupon_bond

.. code-block:: python

   import torch
   from torchquantlib import zero_coupon_bond

   price = zero_coupon_bond(torch.tensor(1000.0), torch.tensor(0.05), torch.tensor(5.0))
   print(price)

Coupon Bond
-----------

.. autofunction:: torchquantlib.core.asset_pricing.bond.bond_pricer.coupon_bond

.. code-block:: python

   import torch
   from torchquantlib import coupon_bond

   price = coupon_bond(torch.tensor(1000.0), torch.tensor(0.06), torch.tensor(0.05), torch.tensor(10))
   print(price)

Callable / Putable / Convertible / Stochastic Rate Bonds
--------------------------------------------------------

.. autofunction:: torchquantlib.core.asset_pricing.bond.bond_pricer.callable_bond
.. autofunction:: torchquantlib.core.asset_pricing.bond.bond_pricer.putable_bond
.. autofunction:: torchquantlib.core.asset_pricing.bond.bond_pricer.convertible_bond
.. autofunction:: torchquantlib.core.asset_pricing.bond.bond_pricer.stochastic_rate_bond

These helpers are also available directly from ``torchquantlib``.
