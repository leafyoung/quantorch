Heston Model
============

.. automodule:: torchquantlib.models.stochastic_volatility.heston
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Heston
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   from torchquantlib import Heston

   model = Heston(
       kappa_init=2.0,
       theta_init=0.04,
       sigma_v_init=0.3,
       rho_init=-0.7,
       v0_init=0.04,
       mu_init=0.05,
   )
