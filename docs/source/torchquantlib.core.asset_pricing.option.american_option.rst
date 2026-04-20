American Option
===============

.. automodule:: torchquantlib.core.asset_pricing.option.american_option
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The public American option API is the function
``american_option(option_type, spot, strike, expiry, volatility, rate, steps)``.
It uses a binomial tree and an explicit ``steps`` argument, which matches the
supported examples and tests.

Example
-------

.. code-block:: python

   import torch
   from torchquantlib import american_option

   price = american_option(
       "call",
       torch.tensor(100.0),
       torch.tensor(95.0),
       torch.tensor(1.0),
       torch.tensor(0.2),
       torch.tensor(0.05),
       100,
   )
   print(f"American call option price: {price.item():.4f}")
