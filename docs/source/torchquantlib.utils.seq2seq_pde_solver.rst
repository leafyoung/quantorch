Seq2Seq PDE Solver
==================

.. automodule:: torchquantlib.utils.seq2seq_pde_solver
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: torchquantlib.utils.seq2seq_pde_solver.Seq2SeqPDESolver
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   import torch
   from torchquantlib import Seq2SeqPDESolver

   model = Seq2SeqPDESolver(input_dim=1, hidden_dim=64, output_dim=1, num_layers=2)
   src = torch.randn(8, 20, 1)
   trg = torch.randn(8, 20, 1)
   output = model(src, trg)
   print(output.shape)

Notes
-----

- The current public utility is the ``Seq2SeqPDESolver`` class itself.
- The module does not expose separate ``Encoder`` or ``Decoder`` public classes.
