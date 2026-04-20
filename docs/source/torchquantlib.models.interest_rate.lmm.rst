Libor Market Model (LMM)
========================

.. automodule:: torchquantlib.models.interest_rate.lmm
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LMM
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   import numpy as np
   from torchquantlib import LMM

   model = LMM(
       forward_rates_init=[0.01, 0.015, 0.02],
       volatilities_init=[0.2, 0.18, 0.16],
       correlations_init=np.eye(3),
   )
