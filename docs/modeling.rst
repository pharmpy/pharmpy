========
Modeling
========

While the :py:class:`pharmpy.model.Model` class can be directly manipulated with low level operations the modeling module offers higher level operations and transformations for building a model. These transformations are also available via the Pharmpy command line interface.


.. math::

~~~~~~~~~~~~~~~~~~~~~~~~~
PK models and ODE systems
~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::
   :hide-output:
   :hide-code:

   from pathlib import Path
   path = Path('tests/testdata/nonmem/')

.. jupyter-execute::

   from pharmpy import Model

   model = Model(path / "pheno_real.mod")

The ODE system of a PK model can be converted from having a compartmental description to be described with an explicit ODE-system.

.. jupyter-execute::

   from pharmpy.modeling import explicit_odes

   print(model.statements.ode_system)
   explicit_odes(model)
   print(model.statements.ode_system)

For NONMEM models this means going from any of the compartmental ADVANS (ADVAN1-4, ADVAN10-12) to coding using an explicit $DES.

.. jupyter-execute::

   print(model)
