Scenario Analysis
=================

.. currentmodule:: torchquantlib.core.risk.market_risk.scenario_analysis

Scenario analysis applies simple scenario multipliers to a current portfolio value.

.. autofunction:: scenario_analysis

Example
-------

.. code-block:: python

   import torch
   from torchquantlib import scenario_analysis

   portfolio_value = torch.tensor(100000.0)
   scenarios = torch.tensor([0.9, 1.0, 1.1])

   scenario_values = scenario_analysis(portfolio_value, scenarios)
   print(scenario_values)
