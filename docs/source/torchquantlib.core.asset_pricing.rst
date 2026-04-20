Asset pricing
=============

The asset pricing package re-exports the most commonly used pricing functions.

.. code-block:: python

   from torchquantlib import (
       american_option,
       asian_option,
       bermudan_option,
       black_scholes_merton,
       future_pricer,
       zero_coupon_bond,
   )

.. automodule:: torchquantlib.core.asset_pricing
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :maxdepth: 2

   torchquantlib.core.asset_pricing.bond
   torchquantlib.core.asset_pricing.money
   torchquantlib.core.asset_pricing.option
