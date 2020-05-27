====
FREM
====

Pharmpy currently handles the postprocessing, plotting and creation of model_3b of the PsN FREM tool.

.. math::

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The FREM postprocessing and results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All FREM postprocessing use the results from the final FREM model (model_4) only. Let us denote the FREM matrix :math:`\Omega`. This matrix is then a partition as follows:

.. math::

   \Omega = 

The first step is to find the parameter uncertainty of the parameters of the


.. jupyter-execute::
   :hide-output:
   :hide-code:

   from pathlib import Path
   path = Path('tests/testdata/nonmem/')

.. jupyter-execute::

   from pharmpy import Model

   model = Model(path / "pheno_real.mod")
   df = model.dataset
   df
