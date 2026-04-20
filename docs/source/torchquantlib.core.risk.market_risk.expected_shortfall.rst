Expected Shortfall (ES)
=======================

.. currentmodule:: torchquantlib.core.risk.market_risk.expected_shortfall

Use ``calculate_es`` for Expected Shortfall / CVaR.

.. autofunction:: calculate_es

Example
-------

.. code-block:: python

   import torch
   from torchquantlib import calculate_es

   returns = torch.randn(1000)
   es_95 = calculate_es(returns, confidence_level=0.95)
   print(f"Expected Shortfall (95% confidence): {es_95.item():.4f}")

Notes
-----

- Expected Shortfall complements VaR by averaging tail losses beyond the VaR threshold.
- The implementation reuses the public ``calculate_var`` function internally.
