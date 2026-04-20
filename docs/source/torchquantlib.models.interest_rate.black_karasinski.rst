Black-Karasinski Model
======================

.. automodule:: torchquantlib.models.interest_rate.black_karasinski
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BlackKarasinski
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   from torchquantlib import BlackKarasinski

   model = BlackKarasinski(a_init=0.1, sigma_init=0.01, r0_init=0.03)
