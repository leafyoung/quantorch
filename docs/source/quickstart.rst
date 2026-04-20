Quickstart Guide
================

This guide shows the smallest supported public API entrypoints for option pricing.

Basic usage
-----------

.. code-block:: python

   import torch
   from torchquantlib import black_scholes_merton

   spot = torch.tensor(100.0)
   strike = torch.tensor(105.0)
   expiry = torch.tensor(1.0)
   volatility = torch.tensor(0.2)
   rate = torch.tensor(0.05)
   dividend = torch.tensor(0.02)

   price = black_scholes_merton(
       "call",
       "european",
       spot,
       strike,
       expiry,
       volatility,
       rate,
       dividend,
   )
   print(f"Option Price: {price.item()}")

American option pricing
-----------------------

.. code-block:: python

   import torch
   from torchquantlib import american_option

   price = american_option(
       "put",
       torch.tensor(100.0),
       torch.tensor(100.0),
       torch.tensor(1.0),
       torch.tensor(0.2),
       torch.tensor(0.05),
       100,
   )
   print(f"American Option Price: {price.item()}")
