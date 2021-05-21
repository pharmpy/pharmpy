.. _modeling:

========
Modeling
========

While the :py:class:`pharmpy.model.Model` class can be directly manipulated
with low level operations the modeling module offers higher level operations and transformations for building a model.
These transformations are also available via the Pharmpy command line interface. To read more about these functions
such as how the initial estimates of parameters are chosen, see their respective API documentation.

.. jupyter-execute::
   :hide-output:
   :hide-code:

   from pathlib import Path
   path = Path('tests/testdata/nonmem/')
   from docs.help_functions import print_model_diff

The following model is the start model for the examples.

.. jupyter-execute::

   from pharmpy import Model

   model_ref = Model(path / "pheno.mod")
   print(model_ref)

~~~~~~~~~~~~~~~
Basic modelling
~~~~~~~~~~~~~~~

Many basic model manipulation tasks that could also be done using methods on model objects have been included in the modeling module. This
makes it possible to do most common model manipulations using a functional interface that is easy to chain into a pipeline.

.. warning::

   Note that all manipulations are done in place, i.e. the model referenced by the input argument will be changed.

Reading, writing and updating source models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read model from file
====================

.. jupyter-execute::
   :hide-output:

   from pharmpy.modeling import *
   model = read_model(path / 'pheno.mod')

Read model from string
======================

If the model code is in a string variable it can be read in directly.

.. jupyter-execute::
    :hide-output:

    code = '$PROBLEM base model\n$INPUT ID DV TIME\n$DATA file.csv IGNORE=@\n$PRED Y = THETA(1) + ETA(1) + ERR(1)\n$THETA 0.1\n$OMEGA 0.01\n$SIGMA 1\n$ESTIMATION METHOD=1'
    model = read_model_from_string(code)

Update model source
===================

Changes done to a model will not affect the model code (e.g. the NONMEM code) until performing an update of the source.

.. jupyter-execute::
   :hide-output:

   model = Model(path / 'pheno.mod')
   update_source(model)

Write model to file
===================

.. code::

   write_model(model, 'mymodel.mod')

Parameters
~~~~~~~~~~

Fix and unfix parameters
========================

The functions for fixing/unfixing parameters take either a list of parameter names or one single parameter name string.

.. jupyter-execute::
   :hide-output:

   fix_parameters(model, ['THETA(1)', 'THETA(2)'])
   unfix_parameters(model, 'THETA(1)')

It is also possible to fix and unfix the parameters to a specified value or to a list of values. If parameter_names
is None, all parameters will be transformed.

.. jupyter-execute::
   :hide-output:

   fix_parameters_to(model, ['THETA(1)', 'THETA(2)'], [0, 1])
   fix_parameters_to(model, ['THETA(1)', 'THETA(2)'], 0)
   unfix_parameters_to(model, 'THETA(1)', 0)
   unfix_parameters_to(model, None, 0)


Add parameter
=============

A new parameter can be added by using the name of the new parameter.

.. jupyter-execute::

   model = Model(path / 'pheno.mod')
   add_parameter(model, 'MAT')
   update_source(model)
   print_model_diff(model_ref, model)

~~~~~~~~~~~~~~~~~~~~~~~~~
PK models and ODE systems
~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   model = Model(path / "pheno.mod")

The ODE system of a PK model can be converted from having a compartmental description to be described with an explicit ODE-system.

.. jupyter-execute::

   from pharmpy.modeling import explicit_odes

   print(model.statements.ode_system)
   explicit_odes(model)
   print(model.statements.ode_system)

For NONMEM models this means going from any of the compartmental ADVANs (ADVAN1-4, ADVAN10-12) to coding using an explicit $DES. The exact solver to use (i.e. the specific ADVAN to use) can be set using the function `set_ode_solver`.

.. jupyter-execute::

   set_ode_solver(model, 'ADVAN13')
   model.update_source()
   print_model_diff(model_ref, model)

Absorption rate
~~~~~~~~~~~~~~~

Bolus absorption
==================

Let us use a model with bolus absorption as a starting point.

.. graphviz::

   digraph fo {
     rankdir = LR
     node [shape=box]
     S [label="S", style=invis, width=0, height=0, margin=0];
     Output [label="O", style=invis, width=0, height=0, margin=0];
     "Central" -> Output [label=K];
     S -> "Central" [label="Bolus"];
   }

.. jupyter-execute::

   from pharmpy.modeling import bolus_absorption
   model = Model(path / "pheno.mod")

This type of absorption can be created with:

.. jupyter-execute::

    bolus_absorption(model)
    model.update_source()
    print_model_diff(model_ref, model)


Zero order
===========

Let us now change to zero order absorption.

.. graphviz::

   digraph fo {
     rankdir = LR
     node [shape=box]
     S [label="S", style=invis, width=0, height=0, margin=0];
     Output [label="O", style=invis, width=0, height=0, margin=0];
     "Central" -> Output [label=K];
     S -> "Central" [label=Infusion];
   }

See :py:func:`pharmpy.modeling.zero_order_absorption`.

.. jupyter-execute::

   from pharmpy.modeling import zero_order_absorption
   zero_order_absorption(model)
   model.update_source(nofiles=True)
   print_model_diff(model_ref, model)

First order
===========

First order absorption would mean adding an absorption (depot) compartment like this:

.. graphviz::

   digraph fo {
     rankdir = LR
     node [shape=box]
     S [label="S", style=invis, width=0, height=0, margin=0];
     Output [label="O", style=invis, width=0, height=0, margin=0];
     "Depot" -> "Central" [label=Ka];
     "Central" -> Output [label=K];
     S -> "Depot" [label=Bolus];
   }

See :py:func:`pharmpy.modeling.first_order_absorption`.

.. jupyter-execute::

   from pharmpy.modeling import first_order_absorption
   first_order_absorption(model)
   model.update_source(nofiles=True)
   print_model_diff(model_ref, model)

Sequential zero-order then first-order
======================================

Sequential zero-order absorption followed by first-order absorption will have an infusion dose into the depot compartment.

.. graphviz::

   digraph fo {
     rankdir = LR
     node [shape=box]
     S [label="S", style=invis, width=0, height=0, margin=0];
     Output [label="O", style=invis, width=0, height=0, margin=0];
     "Depot" -> "Central" [label=Ka];
     "Central" -> Output [label=K];
     S -> "Depot" [label=Infusion];
   }

See :py:func:`pharmpy.modeling.seq_zo_fo_absorption`.

.. jupyter-execute::

   from pharmpy.modeling import seq_zo_fo_absorption
   seq_zo_fo_absorption(model)
   model.update_source(nofiles=True)
   print_model_diff(model_ref, model)

Absorption delay
~~~~~~~~~~~~~~~~

Transit compartments
====================

Transit compartments can be added or removed using the :py:func:`pharmpy.modeling.set_transit_compartments` function.

.. jupyter-execute::

   model = Model(path / "pheno.mod")
   from pharmpy.modeling import set_transit_compartments

   set_transit_compartments(model, 4)
   model.update_source()
   print_model_diff(model_ref, model)


Lag time
========

.. jupyter-execute::

   model = Model(path / "pheno.mod")

Lag time may be added to a dose compartment of a model.

.. jupyter-execute::

   from pharmpy.modeling import add_lag_time
   add_lag_time(model)
   model.update_source()
   print_model_diff(model_ref, model)

Similarly, to remove lag time:

.. jupyter-execute::

   from pharmpy.modeling import remove_lag_time
   remove_lag_time(model)
   model.update_source()
   print_model_diff(model_ref, model)

Elimination rate
~~~~~~~~~~~~~~~~

Pharmpy supports changing a model to first-order, zero-order, Michaelis-Menten, and first-order + Michaelis-Menten
elimination.

First-order elimination
=======================

.. jupyter-execute::

   from pharmpy.modeling import first_order_elimination
   model = Model(path / "pheno.mod")
   first_order_elimination(model)
   model.update_source()
   print_model_diff(model_ref, model)

See :py:func:`pharmpy.modeling.first_order_elimination`.

Zero-order elimination
======================

.. jupyter-execute::

   from pharmpy.modeling import zero_order_elimination
   model = Model(path / "pheno.mod")
   zero_order_elimination(model)
   model.update_source()
   print_model_diff(model_ref, model)

See :py:func:`pharmpy.modeling.zero_order_elimination`.

Michaelis-Menten elimination
============================

.. jupyter-execute::

   from pharmpy.modeling import michaelis_menten_elimination
   model = Model(path / "pheno.mod")
   michaelis_menten_elimination(model)
   model.update_source()
   print_model_diff(model_ref, model)

See :py:func:`pharmpy.modeling.michaelis_menten_elimination`.

Mixed Michaelis-Menten + First-Order elimination
===================================================

.. jupyter-execute::

   from pharmpy.modeling import mixed_mm_fo_elimination
   model = Model(path / "pheno.mod")
   mixed_mm_fo_elimination(model)
   model.update_source()
   print_model_diff(model_ref, model)

See :py:func:`pharmpy.modeling.mixed_mm_fo_elimination`.

Distribution
~~~~~~~~~~~~

Add peripheral compartment
==========================

.. jupyter-execute::

   model = Model(path / "pheno.mod")

Adding a peripheral compartment.

.. jupyter-execute::

   from pharmpy.modeling import add_peripheral_compartment
   add_peripheral_compartment(model)
   model.update_source()
   print_model_diff(model_ref, model)


Remove peripheral compartment
=============================

Removing a peripheral compartment.

.. jupyter-execute::

   from pharmpy.modeling import remove_peripheral_compartment
   remove_peripheral_compartment(model)
   remove_ref = model.copy()
   model.update_source()
   print_model_diff(remove_ref, model)

.. _cov_effects:


Set the number of peripheral compartments
=========================================

As an alternative to adding or removing one peripheral compartment a certain number of peripheral compartents can be set directly.

.. jupyter-execute::

   from pharmpy.modeling import set_peripheral_compartments
   set_peripheral_compartments(model, 2)
   remove_ref = model.copy()
   model.update_source()
   print_model_diff(remove_ref, model)



~~~~~~~~~~~~~~~~~~~~~~~~
Adding covariate effects
~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   model = Model(path / "pheno.mod")

Covariate effects may be applied to a model.

.. jupyter-execute::
   :hide-output:

   from pharmpy.modeling import add_covariate_effect
   add_covariate_effect(model, 'CL', 'WGT', 'pow', operation='*')

Here, *CL* indicates the name of the parameter onto which you want to apply the effect, *WGT* is the name of the
covariate, and *pow* (power function) is the effect you want to apply. The effect can be either
added or multiplied to the parameter, denoted by '*' or '+' (multiplied is default).

.. jupyter-execute::

   model.update_source()
   print_model_diff(model_ref, model)

.. note::

   To see the list of available effects and how the initial estimates for each type of effect is chosen,
   see :py:class:`pharmpy.modeling.add_covariate_effect`.

Pharmpy also supports user formatted covariate effects.

.. jupyter-execute::
   :hide-output:

   model = Model(path / "pheno.mod")
   user_effect = '((cov/std) - median) * theta'
   add_covariate_effect(model, 'CL', 'WGT', user_effect, operation='*')

The covariate is denoted as *cov*, the theta as *theta* (or, if multiple thetas: *theta1*, *theta2* etc.), and the mean,
median, and standard deviation as *mean*, *median*, and *std* respectively. This is in order for
the names to be substituted with the correct symbols.

.. jupyter-execute::

   model.update_source()
   print_model_diff(model_ref, model)

~~~~~~~~~~~~~~~~~~~~~~
Transformation of etas
~~~~~~~~~~~~~~~~~~~~~~

Boxcox
~~~~~~

.. jupyter-execute::

   model = Model(path / "pheno.mod")

To apply a boxcox transformation, input a list of the etas of interest. See
:py:func:`pharmpy.modeling.boxcox`.

.. jupyter-execute::

   from pharmpy.modeling import boxcox
   boxcox(model, ['ETA(1)'])
   model.update_source()
   print_model_diff(model_ref, model)

This can be done for one or multiple etas. The new model will have new statements where *ETAB1* is a boxcox
transformation of *ETA(1)*.

If no list is provided, all etas will be updated.

.. jupyter-execute::

   model = Model(path / "pheno.mod")
   boxcox(model)
   model.update_source()
   print_model_diff(model_ref, model)

Approximate t-distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~

Applying an approximate t-distribution transformation of etas is analogous to a boxcox transformation. The input
is a list of etas, and if no list is provided all etas will be transformed. See
:py:func:`pharmpy.modeling.tdist`.

.. jupyter-execute::

   model = Model(path / "pheno.mod")
   from pharmpy.modeling import tdist
   tdist(model, ['ETA(1)'])
   model.update_source()
   print_model_diff(model_ref, model)

John Draper
~~~~~~~~~~~

John Draper transformation is also supported. The function takes a list of etas as input, if no list is
provided all etas will be transformed. See :py:func:`pharmpy.modeling.john_draper`.

.. jupyter-execute::

   model = Model(path / "pheno.mod")
   from pharmpy.modeling import john_draper
   john_draper(model, ['ETA(1)'])
   model.update_source()
   print_model_diff(model_ref, model)

~~~~~~~~~~~~~~~
Adding new etas
~~~~~~~~~~~~~~~

Adding IIVs
~~~~~~~~~~~

.. jupyter-execute::

   model = Model(path / "pheno.mod")

IIVs may be added to a model.

.. jupyter-execute::
   :hide-output:

   from pharmpy.modeling import add_iiv
   add_iiv(model, 'S1', 'exp', operation='*')

In this example, *S1* is the parameter to add the IIV to, *exp* is the effect on the new eta (see
:py:class:`pharmpy.modeling.add_iiv` for available templates and how initial estimates are chosen). The
operation denotes whether the new eta should be added or multiplied (default).

.. jupyter-execute::

   model.update_source()
   print_model_diff(model_ref, model)

For some of the templates, such as proportional etas, the operation can be omitted since it is
already defined by the effect.

.. jupyter-execute::

   model = Model(path / "pheno.mod")
   add_iiv(model, 'S1', 'prop')
   model.update_source()
   print_model_diff(model_ref, model)

A list of parameter names can also be used as input. In that case, the effect and the operation (if not omitted) must
be either a string (in that case, all new IIVs will have those settings) or be a list of the same size.

.. jupyter-execute::

   model = Model(path / "pheno.mod")
   add_iiv(model, ['V', 'S1'], 'exp')
   model.update_source()
   print_model_diff(model_ref, model)


Similarly to when you :ref:`add a covariate effect<cov_effects>`, you can add user
specified effects.

.. jupyter-execute::
   :hide-output:

   model = Model(path / "pheno.mod")
   user_effect = 'eta_new**2'
   add_iiv(model, 'S1', user_effect, operation='*')

The new etas need to be denoted as *eta_new*.

.. jupyter-execute::

   model.update_source()
   print_model_diff(model_ref, model)

You can also provide a custom eta name, i.e the name of the internal representation of the eta in Pharmpy. For
example, if you want to be able to use the NONMEM name.

.. jupyter-execute::

   model = Model(path / "pheno.mod")
   add_iiv(model, 'S1', 'exp', eta_names='ETA(3)')
   model.update_source()
   model.random_variables


Adding IOVs
~~~~~~~~~~~

.. jupyter-execute::

   model = Model(path / "pheno.mod")

.. jupyter-execute::
   :hide-output:
   :hide-code:

   import numpy as np
   model.dataset['FA1'] = np.random.randint(0, 2, len(model.dataset.index))

Similarly, you can also add IOVs to your model.

.. jupyter-execute::
   :hide-output:

   from pharmpy.modeling import add_iov
   add_iov(model, 'FA1', ['ETA(1)'])

In this example, *FA1* is the name of the occasion column, and the etas on which you wish to add the IOV on are
provided as a list. See :py:class:`pharmpy.modeling.add_iov` for information on how initial estimates are chosen.

.. jupyter-execute::

   model.update_source()
   print_model_diff(model_ref, model)

The name of the parameter may also be provided as an argument, and a mix of eta names and parameter names is
supported.

.. jupyter-execute::

   model = Model(path / "pheno.mod")

.. jupyter-execute::
   :hide-output:
   :hide-code:

   model.dataset['FA1'] = np.random.randint(0, 2, len(model.dataset.index))

.. jupyter-execute::

   add_iov(model, 'FA1', ['CL', 'ETA(2)'])
   model.update_source()
   print_model_diff(model_ref, model)

.. _add_iov_custom_names:

Custom eta names are supported, meaning that the internal representation of the eta in Pharmpy can be set via
the eta_names argument. For example, if you want to be able to use the NONMEM name.

.. warning::
   The number of names must be equal to the number of created etas (i.e. the number of
   input etas times the number of categories for occasion).

.. jupyter-execute::

   model = Model(path / "pheno.mod")

.. jupyter-execute::
   :hide-output:
   :hide-code:

   model.dataset['FA1'] = np.random.randint(0, 2, len(model.dataset.index))

.. jupyter-execute::

   add_iov(model, 'FA1', ['ETA(1)'], eta_names=['ETA(3)', 'ETA(4)'])
   model.update_source()
   model.random_variables


~~~~~~~~~~~~~
Removing etas
~~~~~~~~~~~~~

Remove IIVs
~~~~~~~~~~~

.. jupyter-execute::

   model = Model(path / "pheno.mod")

Etas can also be removed by providing a list of etas and/or name of parameters to remove IIV from. See
:py:func:`pharmpy.modeling.remove_iiv`.

.. jupyter-execute::

   from pharmpy.modeling import remove_iiv
   remove_iiv(model, ['ETA(1)', 'V'])
   model.update_source()
   print_model_diff(model_ref, model)

If you want to remove all etas, leave argument empty.

.. jupyter-execute::

   model = Model(path / "pheno.mod")
   from pharmpy.modeling import remove_iiv
   remove_iiv(model)
   model.update_source()
   print_model_diff(model_ref, model)

Remove IOVs
~~~~~~~~~~~

You can remove IOVs as well, however all IOV omegas will be removed. See
:py:func:`pharmpy.modeling.remove_iov`.

.. jupyter-execute::
   :hide-output:
   :hide-code:

    import warnings
    warnings.filterwarnings('ignore', message='No IOVs present')

.. jupyter-execute::
   :hide-output:

   model = Model(path / "pheno.mod")
   from pharmpy.modeling import remove_iov
   remove_iov(model)
   model.update_source()

~~~~~~~~~~~~~~~
The error model
~~~~~~~~~~~~~~~

Removing the error model
~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
   Removing all epsilons might lead to a model that isn't runnable.

.. jupyter-execute::
   :hide-output:

   model = Model(path / "pheno.mod")

The error model can be removed.

.. jupyter-execute::

   from pharmpy.modeling import remove_error

   remove_error(model)
   model.update_source()
   print_model_diff(model_ref, model)

Setting an additive error model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The additive error model is :math:`y = f + \epsilon_a`. In the case of log transformed data the
same error model can be approximated to :math:`y = \log f + \frac{\epsilon_a}{f}`. This because

.. math::

    \log (f + \epsilon_a) = \log (f(1+\frac{\epsilon_a}{f})) = \log f + \log(1 + \frac{\epsilon_a}{f}) \approx \log f + \frac{\epsilon_a}{f}

where the approximation is the first term of the Taylor expansion of :math:`\log(1 + x)`.


.. jupyter-execute::
   :hide-output:

   model = Model(path / "pheno.mod")

To set an additive error model:

.. jupyter-execute::

   from pharmpy.modeling import additive_error

   additive_error(model)
   model.statements.find_assignment('Y')

.. jupyter-execute::

   model.update_source()
   print_model_diff(model_ref, model)

To set an additive error model with log transformed data:


.. jupyter-execute::

   from pharmpy.modeling import additive_error

   model = Model(path / "pheno.mod")
   additive_error(model, data_trans='log(Y)')
   model.update_source()
   print_model_diff(model_ref, model)

or set the `data_transformation` attribute on the model.

See :py:func:`pharmpy.modeling.additive_error`.

Setting a proportional error model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The proportinal error model is :math:`y = f + f \epsilon_p`. In the case of log transformed data the
same error model can be approximated to :math:`y = \log f + \epsilon_p`. This because

.. math::

    \log (f + f\epsilon_p) = \log (f(1+\epsilon_p)) = \log f + \log(1+ \epsilon_p) \approx \log f + \epsilon_p

where again the approximation is the first term of the Taylor expansion of :math:`\log(1 + x)`.

.. jupyter-execute::
   :hide-output:

   model = Model(path / "pheno.mod")

To set a proportional error model:

.. jupyter-execute::

   from pharmpy.modeling import proportional_error

   proportional_error(model)
   model.statements.find_assignment('Y')

.. jupyter-execute::

   model.update_source()
   print_model_diff(model_ref, model)

To set a proportional error model with log transformed data:

.. jupyter-execute::

   from pharmpy.modeling import proportional_error

   model = Model(path / "pheno.mod")
   proportional_error(model, data_trans='log(Y)')
   model.update_source()
   print_model_diff(model_ref, model)


See :py:func:`pharmpy.modeling.proportional_error`.

Setting a combined additive and proportional error model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The combined error model is :math:`y = f + f \epsilon_p + \epsilon_a`. In the case of log transformed data the
same error model can be approximated to :math:`y = \log f + \epsilon_p + \frac{\epsilon_a}{f}`. This because

.. math::

    \log (f + f\epsilon_p + \epsilon_a) = \log (f(1+\epsilon_p+\frac{\epsilon_a}{f})) = \log f + \log(1 + \epsilon_p + \frac{\epsilon_a}{f}) \approx \log f + \epsilon_p + \frac{\epsilon_a}{f}

where again the approximation is the first term of the Taylor expansion of :math:`\log(1 + x)`.

.. jupyter-execute::
   :hide-output:

   model = Model(path / "pheno.mod")

To set a combined error model:

.. jupyter-execute::

   from pharmpy.modeling import combined_error

   combined_error(model)
   model.statements.find_assignment('Y')

.. jupyter-execute::

   model.update_source()
   print_model_diff(model_ref, model)

To set a combined error model with log transformed data:

.. jupyter-execute::

   from pharmpy.modeling import combined_error

   model = Model(path / "pheno.mod")
   combined_error(model, data_trans='log(Y)')
   model.update_source()
   print_model_diff(model_ref, model)


See :py:func:`pharmpy.modeling.combined_error`.

Applying IIV on RUVs
~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::
   :hide-output:

   model = Model(path / "pheno.mod")

IIVs can be added to RUVs by multiplying epsilons with an exponential new eta.

.. jupyter-execute::

   from pharmpy.modeling import iiv_on_ruv

   iiv_on_ruv(model, ['EPS(1)'])
   model.update_source()
   print_model_diff(model_ref, model)

Input a list of the epsilons you wish to transform, leave argument empty if all epsilons should be
transformed.

.. jupyter-execute::

   model = Model(path / "pheno.mod")
   iiv_on_ruv(model)
   model.update_source()
   print_model_diff(model_ref, model)

See :py:func:`pharmpy.modeling.iiv_on_ruv`.

Custom eta names are supported the same way as when :ref:`adding IOVs<add_iov_custom_names>`.

.. jupyter-execute::

   model = Model(path / "pheno.mod")
   iiv_on_ruv(model, ['EPS(1)'], eta_names=['ETA(3)'])
   model.random_variables


Power effects on RUVs
~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   from pharmpy.modeling import power_on_ruv
   model = Model(path / "pheno.mod")
   power_on_ruv(model, ['EPS(1)'])
   model.update_source()
   print_model_diff(model_ref, model)

A power effect will be applied to all provided epsilons, leave argument empty if all
epsilons should be transformed.

See :py:func:`pharmpy.modeling.power_on_ruv`.

Estimate standard deviation of epsilons with thetas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Someimes it is useful to estimate a theta instead of a sigma. This can be done by fixing the sigma to 1 and multiplying the
correspondng epsilon with a theta. This way the theta will represent the standard deviation of the epsilon.

.. jupyter-execute::

    from pharmpy.modeling import theta_as_stdev
    model = Model(path / "pheno.mod")
    theta_as_stdev(model)
    model.update_source()
    print_model_diff(model_ref, model)

Weighted error model
~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    from pharmpy.modeling import set_weighted_error_model
    model = Model(path / "pheno.mod")
    set_weighted_error_model(model)
    model.update_source()
    print_model_diff(model_ref, model)

dTBS error model
~~~~~~~~~~~~~~~~

.. jupyter-execute::

    from pharmpy.modeling import set_weighted_error_model
    model = Model(path / "pheno.mod")
    set_dtbs_error(model)
    model.update_source(nofiles=True)
    print_model_diff(model_ref, model)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Creating full or partial block structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::
   :hide-output:

   model = Model(path / "pheno.mod")

Pharmpy supports the creation of full and partial block structures of etas. See
:py:func:`pharmpy.modeling.create_rv_block`.

.. jupyter-execute::

   from pharmpy.modeling import create_rv_block

   create_rv_block(model, ['ETA(1)', 'ETA(2)'])
   model.update_source()
   print_model_diff(model_ref, model)

To create a partial block structure, provide the etas as a list. Valid etas must be IIVs and cannot be
fixed. If no list is provided as input, a full block structure is implemented.

.. jupyter-execute::

   model = Model(path / "pheno.mod")
   create_rv_block(model)
   model.update_source()
   print_model_diff(model_ref, model)

.. warning::

   If you have an eta block and wish to include another eta, note that you need to have all etas from that
   block as input argument, any that are not included will be separated from that block.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Remove covariance between etas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::
   :hide-output:

   model = Model(path / "pheno.mod")

Covariance can be removed between etas using the function :py:func:`pharmpy.modeling.split_rv_block`. If we have
the model:

.. jupyter-execute::

   from pharmpy.modeling import copy_model, create_rv_block

   create_rv_block(model)
   model.update_source()
   model_block = copy_model(model)
   print(model)

Provide etas as a list.

.. jupyter-execute::

   from pharmpy.modeling import split_rv_block

   split_rv_block(model, ['ETA(1)'])
   model.update_source()
   print_model_diff(model_block, model)

If no list of etas is provided, all block structures will be split.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Update initial estimates from previous run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If there are results from a previous run, those can be used for initial estimates in your
pharmpy model. See :py:func:`pharmpy.modeling.update_inits`.

.. jupyter-execute::

   model = Model(path / "pheno.mod")
   from pharmpy.modeling import update_inits

   update_inits(model, force_individual_estimates=True)
   model.update_source(nofiles=True)


~~~~~~~~~~~~~~~
Fitting a model
~~~~~~~~~~~~~~~

Pharmpy is designed to be able to do fitting of models to data using different external tools. Currently only NONMEM is supported.

.. code-block:: python

    from pharmpy.modeling import fit
    fit(model)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Getting results from a PsN run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pharmpy can create results objects from PsN run directories for some of the PsN tools. The result objects is a collection of different
results from the tool and can be saved as either json or csv.

.. code-block:: python

    from pharmpy.modeling import create_results
    res = create_results("bootstrap_dir1")
    res.to_json("bootstrap_dir1/results.json")
    res.to_csv("bootstrap_dir1/results.csv")
