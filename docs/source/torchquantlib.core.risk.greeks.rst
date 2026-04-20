Greeks
======

TorchQuant exposes Malliavin-calculus-based Greeks through the ``MalliavinGreeks`` class.

.. automodule:: torchquantlib.core.risk.greeks
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MalliavinGreeks
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   import torch
   from torchquantlib import MalliavinGreeks

   greeks = MalliavinGreeks()
   results = greeks.european_option_greeks(
       S0=torch.tensor(100.0),
       K=torch.tensor(100.0),
       T=torch.tensor(1.0),
       r=torch.tensor(0.05),
       sigma=torch.tensor(0.2),
   )
   print(results)

Notes
-----

- The class uses Monte Carlo simulation and can leverage accelerator-backed PyTorch installations.
- Basket, barrier, lookback, digital, and Asian examples are available through the class methods listed above.
