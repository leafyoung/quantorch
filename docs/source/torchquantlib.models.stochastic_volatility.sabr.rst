SABR Model
==========

.. automodule:: torchquantlib.models.stochastic_volatility.sabr
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SABR
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   from torchquantlib import SABR

   model = SABR(alpha_init=0.2, beta_init=0.5, rho_init=0.0, nu_init=0.3, F0=100.0)
