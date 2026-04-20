Vasicek Model
=============

.. automodule:: torchquantlib.models.interest_rate.vasicek
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Vasicek
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   from torchquantlib import Vasicek

   model = Vasicek(kappa_init=0.1, theta_init=0.05, sigma_init=0.01, r0_init=0.03)
