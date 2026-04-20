Yield Curve Construction
========================

.. automodule:: torchquantlib.utils.yield_curve_construction
   :members:
   :undoc-members:
   :show-inheritance:

Bootstrap Yield Curve
---------------------

.. autofunction:: torchquantlib.utils.yield_curve_construction.bootstrap_yield_curve

Example
^^^^^^^

.. code-block:: python

   import torch
   from torchquantlib import bootstrap_yield_curve

   cash_flows = torch.tensor([
       [100, 0, 0],
       [0, 100, 0],
       [0, 0, 100],
   ])
   prices = torch.tensor([98, 95, 90])

   yields = bootstrap_yield_curve(cash_flows, prices)
   print(yields)

Nelson-Siegel Yield Curve
-------------------------

.. autofunction:: torchquantlib.utils.yield_curve_construction.nelson_siegel_yield_curve

Example
^^^^^^^

.. code-block:: python

   import torch
   from torchquantlib import nelson_siegel_yield_curve

   tau = torch.tensor([0.5, 1, 2, 3, 5, 10])
   beta0 = torch.tensor(0.03)
   beta1 = torch.tensor(-0.02)
   beta2 = torch.tensor(0.01)

   yields = nelson_siegel_yield_curve(tau, beta0, beta1, beta2)
   print(yields)
