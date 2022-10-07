.. _modeling:

========
Modeling
========

While the :py:class:`pharmpy.model.Model` class can be directly manipulated
with low level operations the modeling module offers higher level operations and transformations for building a model.
These transformations are also available via the Pharmpy command line interface. To read more about these functions
such as how the initial estimates of parameters are chosen, see their respective API documentation.

.. pharmpy-execute::
   :hide-output:
   :hide-code:

   from pathlib import Path
   path = Path('tests/testdata/nonmem/')
   from docs.help_functions import print_model_diff

The following model is the start model for the examples.

.. pharmpy-execute::

   from pharmpy.modeling import read_model

   model_ref = read_model(path / "pheno.mod")
   print(model_ref)

~~~~~~~~~~~~~~~
Basic modeling
~~~~~~~~~~~~~~~

Many basic model manipulation tasks that could also be done using methods on model objects have been included in the modeling module. This
makes it possible to do most common model manipulations using a functional interface that is easy to chain into a pipeline.

.. warning::

   Note that all manipulations are done in place, i.e. the model referenced by the input argument will be changed.

Reading, writing and updating source models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read model from file
====================

.. pharmpy-execute::
   :hide-output:

   from pharmpy.modeling import *
   model = read_model(path / 'pheno.mod')

Read model from string
======================

If the model code is in a string variable it can be read in directly.

.. pharmpy-execute::
    :hide-output:

    code = '$PROBLEM base model\n$INPUT ID DV TIME\n$DATA file.csv IGNORE=@\n$PRED Y = THETA(1) + ETA(1) + ERR(1)\n$THETA 0.1\n$OMEGA 0.01\n$SIGMA 1\n$ESTIMATION METHOD=1'
    model = read_model_from_string(code)

Getting the model code
======================

The model code (e.g. the NONMEM code) can be retrieved using the `model_code` attribute.

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / 'pheno.mod')
   model.model_code

Write model to file
===================

.. pharmpy-code::

   write_model(model, 'mymodel.mod')

Parameters
~~~~~~~~~~

Fix and unfix parameters
========================

The functions for fixing/unfixing parameters take either a list of parameter names or one single parameter name string.

.. pharmpy-execute::
   :hide-output:

   fix_parameters(model, ['THETA(1)', 'THETA(2)'])
   unfix_parameters(model, 'THETA(1)')

It is also possible to fix and unfix the parameters to a specified value or to a list of values. If parameter_names
is None, all parameters will be transformed.

.. pharmpy-execute::
   :hide-output:

   fix_parameters_to(model, {'THETA(1)': 0, 'THETA(2)': 1})
   fix_parameters_to(model, {'THETA(1)': 0, 'THETA(2)': 0})
   unfix_parameters_to(model, {'THETA(1)': 0})


Add parameter
=============

A new parameter can be added by using the name of the new parameter.

.. pharmpy-execute::

   model = read_model(path / 'pheno.mod')
   add_individual_parameter(model, 'MAT')
   print_model_diff(model_ref, model)

~~~~~~~~~~~~~~~~~~~~~~~~~
PK models and ODE systems
~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. pharmpy-execute::
   :hide-output:

   from pharmpy.modeling import set_bolus_absorption
   model = read_model(path / "pheno.mod")

This type of absorption can be created with:

.. pharmpy-execute::

    set_bolus_absorption(model)
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

See :py:func:`pharmpy.modeling.set_zero_order_absorption`.

.. pharmpy-execute::

   from pharmpy.modeling import set_zero_order_absorption
   set_zero_order_absorption(model)
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

See :py:func:`pharmpy.modeling.set_first_order_absorption`.

.. pharmpy-execute::

   from pharmpy.modeling import set_first_order_absorption
   set_first_order_absorption(model)
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

See :py:func:`pharmpy.modeling.set_seq_zo_fo_absorption`.

.. pharmpy-execute::

   from pharmpy.modeling import set_seq_zo_fo_absorption
   set_seq_zo_fo_absorption(model)
   print_model_diff(model_ref, model)

Absorption delay
~~~~~~~~~~~~~~~~

Transit compartments
====================

Transit compartments can be added or removed using the :py:func:`pharmpy.modeling.set_transit_compartments` function.

.. pharmpy-execute::

   model = read_model(path / "pheno.mod")
   from pharmpy.modeling import set_transit_compartments

   set_transit_compartments(model, 4)
   print_model_diff(model_ref, model)


Lag time
========

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")

Lag time may be added to a dose compartment of a model.

.. pharmpy-execute::

   from pharmpy.modeling import add_lag_time
   add_lag_time(model)
   print_model_diff(model_ref, model)

Similarly, to remove lag time:

.. pharmpy-execute::

   from pharmpy.modeling import remove_lag_time
   remove_lag_time(model)
   print_model_diff(model_ref, model)

Elimination rate
~~~~~~~~~~~~~~~~

Pharmpy supports changing a model to first-order, zero-order, Michaelis-Menten, and first-order + Michaelis-Menten
elimination.

First-order elimination
=======================

.. pharmpy-execute::

   from pharmpy.modeling import set_first_order_elimination
   model = read_model(path / "pheno.mod")
   set_first_order_elimination(model)
   print_model_diff(model_ref, model)

See :py:func:`pharmpy.modeling.set_first_order_elimination`.

Zero-order elimination
======================

.. pharmpy-execute::

   from pharmpy.modeling import set_zero_order_elimination
   model = read_model(path / "pheno.mod")
   set_zero_order_elimination(model)
   print_model_diff(model_ref, model)

See :py:func:`pharmpy.modeling.set_zero_order_elimination`.

Michaelis-Menten elimination
============================

.. pharmpy-execute::

   from pharmpy.modeling import set_michaelis_menten_elimination
   model = read_model(path / "pheno.mod")
   set_michaelis_menten_elimination(model)
   print_model_diff(model_ref, model)

See :py:func:`pharmpy.modeling.set_michaelis_menten_elimination`.

Mixed Michaelis-Menten + First-Order elimination
===================================================

.. pharmpy-execute::

   from pharmpy.modeling import set_mixed_mm_fo_elimination
   model = read_model(path / "pheno.mod")
   set_mixed_mm_fo_elimination(model)
   print_model_diff(model_ref, model)

See :py:func:`pharmpy.modeling.set_mixed_mm_fo_elimination`.

Distribution
~~~~~~~~~~~~

Add peripheral compartment
==========================

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")

Adding a peripheral compartment.

.. pharmpy-execute::

   from pharmpy.modeling import add_peripheral_compartment
   add_peripheral_compartment(model)
   print_model_diff(model_ref, model)


Remove peripheral compartment
=============================

Removing a peripheral compartment.

.. pharmpy-execute::

   from pharmpy.modeling import remove_peripheral_compartment
   remove_peripheral_compartment(model)
   remove_ref = model.copy()
   print_model_diff(remove_ref, model)

.. _cov_effects:


Set the number of peripheral compartments
=========================================

As an alternative to adding or removing one peripheral compartment a certain number of peripheral compartents can be set directly.

.. pharmpy-execute::

   from pharmpy.modeling import set_peripheral_compartments
   set_peripheral_compartments(model, 2)
   remove_ref = model.copy()
   print_model_diff(remove_ref, model)



~~~~~~~~~~~~~~~~~~~~~~~~
Adding covariate effects
~~~~~~~~~~~~~~~~~~~~~~~~

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")

Covariate effects may be applied to a model.

.. pharmpy-execute::
   :hide-output:

   from pharmpy.modeling import add_covariate_effect
   add_covariate_effect(model, 'CL', 'WGT', 'pow', operation='*')

Here, *CL* indicates the name of the parameter onto which you want to apply the effect, *WGT* is the name of the
covariate, and *pow* (power function) is the effect you want to apply. The effect can be either
added or multiplied to the parameter, denoted by '*' or '+' (multiplied is default).

.. pharmpy-execute::

   print_model_diff(model_ref, model)

.. note::

   To see the list of available effects and how the initial estimates for each type of effect is chosen,
   see :py:class:`pharmpy.modeling.add_covariate_effect`.

Pharmpy also supports user formatted covariate effects.

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")
   user_effect = '((cov/std) - median) * theta'
   add_covariate_effect(model, 'CL', 'WGT', user_effect, operation='*')

The covariate is denoted as *cov*, the theta as *theta* (or, if multiple thetas: *theta1*, *theta2* etc.), and the mean,
median, and standard deviation as *mean*, *median*, and *std* respectively. This is in order for
the names to be substituted with the correct symbols.

.. pharmpy-execute::

   print_model_diff(model_ref, model)

~~~~~~~~~~~~~~~~~~~~~~
Transformation of etas
~~~~~~~~~~~~~~~~~~~~~~

Boxcox
~~~~~~

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")

To apply a boxcox transformation, input a list of the etas of interest. See
:py:func:`pharmpy.modeling.transform_etas_boxcox`.

.. pharmpy-execute::

   from pharmpy.modeling import transform_etas_boxcox
   transform_etas_boxcox(model, ['ETA(1)'])
   print_model_diff(model_ref, model)

This can be done for one or multiple etas. The new model will have new statements where *ETAB1* is a boxcox
transformation of *ETA(1)*.

If no list is provided, all etas will be updated.

.. pharmpy-execute::

   model = read_model(path / "pheno.mod")
   transform_etas_boxcox(model)
   print_model_diff(model_ref, model)

Approximate t-distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~

Applying an approximate t-distribution transformation of etas is analogous to a boxcox transformation. The input
is a list of etas, and if no list is provided all etas will be transformed. See
:py:func:`pharmpy.modeling.transform_etas_tdist`.

.. pharmpy-execute::

   model = read_model(path / "pheno.mod")
   from pharmpy.modeling import transform_etas_tdist
   transform_etas_tdist(model, ['ETA(1)'])
   print_model_diff(model_ref, model)

John Draper
~~~~~~~~~~~

John Draper transformation is also supported. The function takes a list of etas as input, if no list is
provided all etas will be transformed. See :py:func:`pharmpy.modeling.transform_etas_john_draper`.

.. pharmpy-execute::

   model = read_model(path / "pheno.mod")
   from pharmpy.modeling import transform_etas_john_draper
   transform_etas_john_draper(model, ['ETA(1)'])
   print_model_diff(model_ref, model)

~~~~~~~~~~~~~~~
Adding new etas
~~~~~~~~~~~~~~~

Adding IIVs
~~~~~~~~~~~

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")

IIVs may be added to a model.

.. pharmpy-execute::
   :hide-output:

   from pharmpy.modeling import add_iiv
   add_iiv(model, 'S1', 'exp', operation='*')

In this example, *S1* is the parameter to add the IIV to, *exp* is the effect on the new eta (see
:py:class:`pharmpy.modeling.add_iiv` for available templates and how initial estimates are chosen). The
operation denotes whether the new eta should be added or multiplied (default).

.. pharmpy-execute::

   print_model_diff(model_ref, model)

For some of the templates, such as proportional etas, the operation can be omitted since it is
already defined by the effect.

.. pharmpy-execute::

   model = read_model(path / "pheno.mod")
   add_iiv(model, 'S1', 'prop')
   print_model_diff(model_ref, model)

A list of parameter names can also be used as input. In that case, the effect and the operation (if not omitted) must
be either a string (in that case, all new IIVs will have those settings) or be a list of the same size.

.. pharmpy-execute::

   model = read_model(path / "pheno.mod")
   add_iiv(model, ['V', 'S1'], 'exp')
   print_model_diff(model_ref, model)


Similarly to when you :ref:`add a covariate effect<cov_effects>`, you can add user
specified effects.

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")
   user_effect = 'eta_new**2'
   add_iiv(model, 'S1', user_effect, operation='*')

The new etas need to be denoted as *eta_new*.

.. pharmpy-execute::

   print_model_diff(model_ref, model)

You can also provide a custom eta name, i.e the name of the internal representation of the eta in Pharmpy. For
example, if you want to be able to use the NONMEM name.

.. pharmpy-execute::

   model = read_model(path / "pheno.mod")
   add_iiv(model, 'S1', 'exp', eta_names='ETA(3)')
   model.random_variables


Adding IOVs
~~~~~~~~~~~

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")

.. pharmpy-execute::
   :hide-output:
   :hide-code:

   import numpy as np
   model.dataset['FA1'] = np.random.randint(0, 2, len(model.dataset.index))

Similarly, you can also add IOVs to your model.

.. pharmpy-execute::
   :hide-output:

   from pharmpy.modeling import add_iov
   add_iov(model, 'FA1', ['ETA(1)'])

In this example, *FA1* is the name of the occasion column, and the etas on which you wish to add the IOV on are
provided as a list. See :py:class:`pharmpy.modeling.add_iov` for information on how initial estimates are chosen.

.. pharmpy-execute::

   print_model_diff(model_ref, model)

The name of the parameter may also be provided as an argument, and a mix of eta names and parameter names is
supported.

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")

.. pharmpy-execute::
   :hide-output:
   :hide-code:

   model.dataset['FA1'] = np.random.randint(0, 2, len(model.dataset.index))

.. pharmpy-execute::

   add_iov(model, 'FA1', ['CL', 'ETA(2)'])
   print_model_diff(model_ref, model)

.. _add_iov_custom_names:

Custom eta names are supported, meaning that the internal representation of the eta in Pharmpy can be set via
the eta_names argument. For example, if you want to be able to use the NONMEM name.

.. warning::
   The number of names must be equal to the number of created etas (i.e. the number of
   input etas times the number of categories for occasion).

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")

.. pharmpy-execute::
   :hide-output:
   :hide-code:

   model.dataset['FA1'] = np.random.randint(0, 2, len(model.dataset.index))

.. pharmpy-execute::

   add_iov(model, 'FA1', ['ETA(1)'], eta_names=['ETA(3)', 'ETA(4)'])
   model.random_variables


~~~~~~~~~~~~~
Removing etas
~~~~~~~~~~~~~

Remove IIVs
~~~~~~~~~~~

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")

Etas can also be removed by providing a list of etas and/or name of parameters to remove IIV from. See
:py:func:`pharmpy.modeling.remove_iiv`.

.. pharmpy-execute::

   from pharmpy.modeling import remove_iiv
   remove_iiv(model, ['ETA(1)', 'V'])
   print_model_diff(model_ref, model)

If you want to remove all etas, leave argument empty.

.. pharmpy-execute::

   model = read_model(path / "pheno.mod")
   from pharmpy.modeling import remove_iiv
   remove_iiv(model)
   print_model_diff(model_ref, model)

Remove IOVs
~~~~~~~~~~~

You can remove IOVs as well, however all IOV omegas will be removed. See
:py:func:`pharmpy.modeling.remove_iov`.

.. pharmpy-execute::
   :hide-output:
   :hide-code:

    import warnings
    warnings.filterwarnings('ignore', message='No IOVs present')

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")
   from pharmpy.modeling import remove_iov
   remove_iov(model)

~~~~~~~~~~~~~~~
The error model
~~~~~~~~~~~~~~~

Removing the error model
~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
   Removing all epsilons might lead to a model that isn't runnable.

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")

The error model can be removed.

.. pharmpy-execute::

   from pharmpy.modeling import remove_error_model

   remove_error_model(model)
   print_model_diff(model_ref, model)

Setting an additive error model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The additive error model is :math:`y = f + \epsilon_a`. In the case of log transformed data the
same error model can be approximated to :math:`y = \log f + \frac{\epsilon_a}{f}`. This because

.. math::

    \log (f + \epsilon_a) = \log (f(1+\frac{\epsilon_a}{f})) = \log f + \log(1 + \frac{\epsilon_a}{f}) \approx \log f + \frac{\epsilon_a}{f}

where the approximation is the first term of the Taylor expansion of :math:`\log(1 + x)`.


.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")

To set an additive error model:

.. pharmpy-execute::

   from pharmpy.modeling import set_additive_error_model

   set_additive_error_model(model)
   model.statements.find_assignment('Y')

.. pharmpy-execute::

   print_model_diff(model_ref, model)

To set an additive error model with log transformed data:


.. pharmpy-execute::

   from pharmpy.modeling import set_additive_error_model

   model = read_model(path / "pheno.mod")
   set_additive_error_model(model, data_trans='log(Y)')
   print_model_diff(model_ref, model)

or set the `data_transformation` attribute on the model.

See :py:func:`pharmpy.modeling.set_additive_error_model`.

Setting a proportional error model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The proportinal error model is :math:`y = f + f \epsilon_p`. In the case of log transformed data the
same error model can be approximated to :math:`y = \log f + \epsilon_p`. This because

.. math::

    \log (f + f\epsilon_p) = \log (f(1+\epsilon_p)) = \log f + \log(1+ \epsilon_p) \approx \log f + \epsilon_p

where again the approximation is the first term of the Taylor expansion of :math:`\log(1 + x)`.

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")

To set a proportional error model:

.. pharmpy-execute::

   from pharmpy.modeling import set_proportional_error_model

   set_proportional_error_model(model)
   model.statements.find_assignment('Y')

.. pharmpy-execute::

   print_model_diff(model_ref, model)

To set a proportional error model with log transformed data:

.. pharmpy-execute::

   from pharmpy.modeling import set_proportional_error_model

   model = read_model(path / "pheno.mod")
   set_proportional_error_model(model, data_trans='log(Y)')
   print_model_diff(model_ref, model)


See :py:func:`pharmpy.modeling.set_proportional_error_model`.

Setting a combined additive and proportional error model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The combined error model is :math:`y = f + f \epsilon_p + \epsilon_a`. In the case of log transformed data the
same error model can be approximated to :math:`y = \log f + \epsilon_p + \frac{\epsilon_a}{f}`. This because

.. math::

    \log (f + f\epsilon_p + \epsilon_a) = \log (f(1+\epsilon_p+\frac{\epsilon_a}{f})) = \log f + \log(1 + \epsilon_p + \frac{\epsilon_a}{f}) \approx \log f + \epsilon_p + \frac{\epsilon_a}{f}

where again the approximation is the first term of the Taylor expansion of :math:`\log(1 + x)`.

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")

To set a combined error model:

.. pharmpy-execute::

   from pharmpy.modeling import set_combined_error_model

   set_combined_error_model(model)
   model.statements.find_assignment('Y')

.. pharmpy-execute::

   print_model_diff(model_ref, model)

To set a combined error model with log transformed data:

.. pharmpy-execute::

   from pharmpy.modeling import set_combined_error_model

   model = read_model(path / "pheno.mod")
   set_combined_error_model(model, data_trans='log(Y)')
   print_model_diff(model_ref, model)


See :py:func:`pharmpy.modeling.set_combined_error_model`.

Applying IIV on RUVs
~~~~~~~~~~~~~~~~~~~~

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")

IIVs can be added to RUVs by multiplying epsilons with an exponential new eta.

.. pharmpy-execute::

   from pharmpy.modeling import set_iiv_on_ruv

   set_iiv_on_ruv(model, ['EPS(1)'])
   print_model_diff(model_ref, model)

Input a list of the epsilons you wish to transform, leave argument empty if all epsilons should be
transformed.

.. pharmpy-execute::

   model = read_model(path / "pheno.mod")
   set_iiv_on_ruv(model)
   print_model_diff(model_ref, model)

See :py:func:`pharmpy.modeling.set_iiv_on_ruv`.

Custom eta names are supported the same way as when :ref:`adding IOVs<add_iov_custom_names>`.

.. pharmpy-execute::

   model = read_model(path / "pheno.mod")
   set_iiv_on_ruv(model, ['EPS(1)'], eta_names=['ETA(3)'])
   model.random_variables


Power effects on RUVs
~~~~~~~~~~~~~~~~~~~~~

.. pharmpy-execute::

   from pharmpy.modeling import set_power_on_ruv
   model = read_model(path / "pheno.mod")
   set_power_on_ruv(model, ['EPS(1)'])
   print_model_diff(model_ref, model)

A power effect will be applied to all provided epsilons, leave argument empty if all
epsilons should be transformed.

See :py:func:`pharmpy.modeling.set_power_on_ruv`.

Estimate standard deviation of epsilons with thetas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Someimes it is useful to estimate a theta instead of a sigma. This can be done by fixing the sigma to 1 and multiplying the
correspondng epsilon with a theta. This way the theta will represent the standard deviation of the epsilon.

.. pharmpy-execute::

    from pharmpy.modeling import use_thetas_for_error_stdev
    model = read_model(path / "pheno.mod")
    use_thetas_for_error_stdev(model)
    print_model_diff(model_ref, model)

Weighted error model
~~~~~~~~~~~~~~~~~~~~

.. pharmpy-execute::

    from pharmpy.modeling import set_weighted_error_model
    model = read_model(path / "pheno.mod")
    set_weighted_error_model(model)
    print_model_diff(model_ref, model)

dTBS error model
~~~~~~~~~~~~~~~~

.. pharmpy-execute::

    from pharmpy.modeling import set_weighted_error_model
    model = read_model(path / "pheno.mod")
    set_dtbs_error_model(model)
    print_model_diff(model_ref, model)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Creating joint distributions of multiple etas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")

Pharmpy supports the joining of multiple etas into a joint distribution. See
:py:func:`pharmpy.modeling.create_joint_distribution`.

.. pharmpy-execute::

   from pharmpy.modeling import create_joint_distribution

   create_joint_distribution(model, ['ETA(1)', 'ETA(2)'])
   print_model_diff(model_ref, model)

The listed etas will be combined into a new distribution. Valid etas must be IIVs and cannot be
fixed. If no list is provided as input, all etas would be included in the same distribution.

.. pharmpy-execute::

   model = read_model(path / "pheno.mod")
   create_joint_distribution(model)
   print_model_diff(model_ref, model)

.. warning::

   If you already have a joint distribution and wish to include another eta, note that you need to have all etas from that
   distribution as input argument, any that are not included will be separated from that distribution.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Remove covariance between etas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. pharmpy-execute::
   :hide-output:

   model = read_model(path / "pheno.mod")

Covariance can be removed between etas using the function :py:func:`pharmpy.modeling.split_joint_distribution`. If we have
the model:

.. pharmpy-execute::

   from pharmpy.modeling import copy_model, create_joint_distribution

   create_joint_distribution(model)
   model_block = copy_model(model)
   print(model)

Provide etas as a list.

.. pharmpy-execute::

   from pharmpy.modeling import split_joint_distribution

   split_joint_distribution(model, ['ETA(1)'])
   print_model_diff(model_block, model)

If no list of etas is provided, all block structures will be split.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Update initial estimates from previous run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If there are results from a previous run, those can be used for initial estimates in your
pharmpy model. See :py:func:`pharmpy.modeling.update_inits`.

.. pharmpy-execute::

   from pharmpy.modeling import read_model, update_inits

   model = read_model(path / "pheno.mod")

   update_inits(model, model.modelfit_results.parameter_estimates)


~~~~~~~~~~~~~~~
Fitting a model
~~~~~~~~~~~~~~~

Pharmpy is designed to be able to do fitting of models to data using different external tools. Currently only NONMEM is supported.

.. pharmpy-code::

    from pharmpy.tools import fit
    fit(model)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Getting results from a PsN run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pharmpy can create results objects from PsN run directories for some of the PsN tools. The result objects is a collection of different
results from the tool and can be saved as either json or csv.

.. pharmpy-code::

    from pharmpy.tools import create_results
    res = create_results("bootstrap_dir1")
    res.to_json("bootstrap_dir1/results.json")
    res.to_csv("bootstrap_dir1/results.csv")

~~~~~~~~~~~~~
Eta shrinkage
~~~~~~~~~~~~~

Eta shrinkage can be calculated either on the standard deviation scale or on the variance scale

.. pharmpy-execute::

    from pharmpy.modeling import calculate_eta_shrinkage

    pe = model.modelfit_results.parameter_estimates
    ie = model.modelfit_results.individual_estimates
    calculate_eta_shrinkage(model, pe, ie)


.. pharmpy-execute::

    calculate_eta_shrinkage(model, pe, ie, sd=True)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Individual parameter calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pharmpy has functions to calculate statistics for individual parameters that are either defined
in the model code or that can be defined expressions containing dataset columns and/or variables
from the model code.

.. pharmpy-code::

    from pharmpy.modeling import calculate_individual_parameter_statistics
    model = read_model(path / 'secondary_parameters'/ 'run2.mod')
