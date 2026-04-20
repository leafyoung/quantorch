Hull-White Model
================

.. automodule:: torchquantlib.models.interest_rate.hull_white
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: HullWhiteModel
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   from torchquantlib import HullWhiteModel

   model = HullWhiteModel(a_init=0.1, sigma_init=0.01, r0_init=0.03)
