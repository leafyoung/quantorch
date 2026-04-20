Value at Risk (VaR)
===================

.. currentmodule:: torchquantlib.core.risk.market_risk.var

Use ``calculate_var`` for historical-simulation VaR.

.. autofunction:: calculate_var

Example
-------

.. code-block:: python

   import torch
   from torchquantlib import calculate_var

   returns = torch.randn(1000)
   var_95 = calculate_var(returns, confidence_level=0.95)
   print(f"Value at Risk (95% confidence): {var_95.item():.4f}")
