Stress Testing
==============

.. currentmodule:: torchquantlib.core.risk.market_risk.stress_testing

Stress testing applies percentage shock scenarios to a current portfolio value.

.. autofunction:: stress_test

Example
-------

.. code-block:: python

   import torch
   from torchquantlib import stress_test

   portfolio_value = torch.tensor(100000.0)
   stress_scenarios = torch.tensor([-0.3, -0.2, -0.1])

   stressed_values = stress_test(portfolio_value, stress_scenarios)
   print(stressed_values)
