Valuation Adjustment API
========================

.. currentmodule:: torchquantlib.core.risk.valuation_adjustment.valuation_adjustment

The valuation adjustment module provides small public helper functions for common
adjustments used in derivatives risk workflows.

.. autofunction:: calculate_cva
.. autofunction:: calculate_dva
.. autofunction:: calculate_fva
.. autofunction:: calculate_mva

Example
-------

.. code-block:: python

   import torch
   from torchquantlib import calculate_cva, calculate_dva, calculate_fva, calculate_mva

   exposure = torch.tensor(100.0)
   default_prob = torch.tensor(0.02)
   recovery_rate = torch.tensor(0.4)
   funding_cost = torch.tensor(0.01)
   funding_spread = torch.tensor(0.005)
   maturity = torch.tensor(1.0)

   print(calculate_cva(exposure, default_prob, recovery_rate))
   print(calculate_dva(exposure, default_prob, recovery_rate))
   print(calculate_fva(exposure, funding_spread, maturity))
   print(calculate_mva(exposure, funding_cost, maturity))
