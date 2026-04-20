Installation
============

Install TorchQuant from PyPI:

.. code-block:: bash

   pip install --upgrade torchquantlib

Install the local project in editable mode:

.. code-block:: bash

   pip install -e .

Optional dependency groups
--------------------------

Install documentation dependencies:

.. code-block:: bash

   pip install -e ".[docs]"

Install test dependencies:

.. code-block:: bash

   pip install -e ".[test]"

Install the common local development toolchain:

.. code-block:: bash

   pip install -e ".[dev]"

Requirements
------------

TorchQuant requires:

- Python 3.7+
- PyTorch
- NumPy
- SciPy
- GeomLoss

For GPU acceleration, ensure that your PyTorch installation matches your CUDA or other accelerator runtime.
