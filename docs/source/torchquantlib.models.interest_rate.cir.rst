Cox-Ingersoll-Ross (CIR) Model
==============================

.. automodule:: torchquantlib.models.interest_rate.cir
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CIR
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   from torchquantlib import CIR

   model = CIR(kappa_init=0.1, theta_init=0.05, sigma_init=0.01, r0_init=0.03)
