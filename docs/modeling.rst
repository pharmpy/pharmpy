.. _modeling:

========
Modeling
========

While the :py:class:`pharmpy.model.Model` class can be directly manipulated
with low level operations the modeling module offers higher level operations and transformations for building a model.
These transformations are also available via the Pharmpy command line interface. To read more about these functions
such as how the initial estimates of parameters are chosen, see their respective :ref:`API documentation<api_ref>`.

.. pharmpy-execute::
   :hide-output:
   :hide-code:

   from pathlib import Path
   path = Path('tests/testdata/nonmem/')
   from docs.help_functions import print_model_diff

~~~~~~~~~~~~~~~
Basic functions
~~~~~~~~~~~~~~~

Reading and writing models
~~~~~~~~~~~~~~~~~~~~~~~~~~

Pharmpy can read in NONMEM models using the :py:func:`pharmpy.modeling.read_model` function:

.. pharmpy-execute::
   :hide-output:

   from pharmpy.modeling import *
   model = read_model(path / 'pheno.mod')

To inspect the read in model code, the :py:func:`pharmpy.modeling.print_model_code` function is available:

.. pharmpy-execute::
   :hide-output:

   print_model_code(model)

If the model code is in a string variable it can be read in directly using
:py:func:`pharmpy.modeling.read_model_from_string`.

.. pharmpy-execute::
    :hide-output:

    code = '$PROBLEM base model\n$INPUT ID DV TIME\n$DATA file.csv IGNORE=@\n$PRED Y = THETA(1) + ETA(1) + ERR(1)\n$THETA 0.1\n$OMEGA 0.01\n$SIGMA 1\n$ESTIMATION METHOD=1'
    model = read_model_from_string(code)

Finally, any Pharmpy model can be written to a file with :py:func:`pharmpy.modeling.write_model`:

.. pharmpy-code::

   write_model(model, 'mymodel.mod')

Loading example models
~~~~~~~~~~~~~~~~~~~~~~

Pharmpy has example models with example datasets that can be accessed using
:py:func:`pharmpy.modeling.load_example_model`:

.. pharmpy-execute::

   model = load_example_model('pheno')
   print_model_code(model)

Converting models
~~~~~~~~~~~~~~~~~

Pharmpy supports the estimation software NONMEM, nlmixr2 and rxODE2, and a Pharmpy model can be converted into those
formats using :py:func:`pharmpy.modeling.convert_model`:

.. pharmpy-execute::
   :hide-output:

   model_nlmixr = convert_model(model, 'nlmixr')

Then we can inspect the new model code:

.. pharmpy-execute::

   print_model_code(model_nlmixr)

.. _basic_models:

~~~~~~~~~~~~~~~~~~~
Create basic models
~~~~~~~~~~~~~~~~~~~

As a starting point for this user guide, we will create a basic PK model using the function
:py:func:`pharmpy.modeling.create_basic_pk_model`:


.. pharmpy-execute::
   :hide-output:

    dataset_path = path / 'pheno.dta'
    model_start = create_basic_pk_model(modeltype='oral',
                                        dataset_path=dataset_path,
                                        cl_init=0.01,
                                        vc_init=1.0,
                                        mat_init=0.1)

We can examine the model statements of the model:

.. pharmpy-execute::
    model_start.statements

We can see that the model is a one compartment model with first order absorption and elimination and no absorption
delay. Examining the random variables:

.. pharmpy-execute::
    model_start.random_variables

Next we can convert the start model from a generic Pharmpy model to a NONMEM model:

.. pharmpy-execute::
   :hide-output:

    model_start = convert_model(model_start, 'nonmem')

We can then examine the NONMEM model code:

.. pharmpy-execute::
    print_model_code(model_start)

~~~~~~~~~~~~~~~~~~~~~~~~
Modeling transformations
~~~~~~~~~~~~~~~~~~~~~~~~

In Pharmpy there are many different modeling functions that modify the model object. In Pharmpy, the model object and
all its attributes are immutable, meaning that the modeling functions always return a copy of the model object.

.. note::

   To see more information on how initial estimates are chosen etc., please check the :ref:`API reference<api_ref>`.

Structural model
~~~~~~~~~~~~~~~~

There are many functions to change or examine the structural model of a PK dataset. Using the
:ref:`base model<basic_models>` from above, we'll go through how to change different aspects of the structural model
step by step.

Absorption rate
===============

As an example, we'll set the absorption of the start model to zero order absorption:

.. pharmpy-execute::

    run1 = set_zero_order_absorption(model_start)
    run1.statements.ode_system

And examine the updated NONMEM code:

.. pharmpy-execute::
    print_model_code(run1)

Note that the ADVAN has been updated.

List of functions to change absorption rate:

* :py:func:`pharmpy.modeling.set_bolus_absorption`
* :py:func:`pharmpy.modeling.set_first_order_absorption`
* :py:func:`pharmpy.modeling.set_seq_zo_fo_absorption`
* :py:func:`pharmpy.modeling.set_zero_order_absorption`

Absorption delay
================

Next, we will add absorption delay.

.. pharmpy-execute::
    run2 = add_lag_time(run1)
    run2.statements.ode_system

And examine the model code:

.. pharmpy-execute::
    print_model_code(run2)

List of functions to change elimination:

* :py:func:`pharmpy.modeling.add_lag_time`
* :py:func:`pharmpy.modeling.remove_lag_time`
* :py:func:`pharmpy.modeling.set_transit_compartments`

Distribution
============

It is possible to change the number of peripheral compartments. Let us add one peripheral compartment

.. pharmpy-execute::

    run3 = add_peripheral_compartment(run2)
    run3.statements.ode_system

And examine the model code:

.. pharmpy-execute::
    print_model_code(run3)

List of functions to change distribution:

* :py:func:`pharmpy.modeling.add_peripheral_compartment`
* :py:func:`pharmpy.modeling.remove_peripheral_compartment`
* :py:func:`pharmpy.modeling.set_peripheral_compartments`

Elimination
===========

Now we will change to non-linear elimination.

.. pharmpy-execute::

    run4 = set_michaelis_menten_elimination(run3)
    run4.statements.ode_system

And examine the model code:

.. pharmpy-execute::
    print_model_code(run4)

List of functions to change elimination:

* :py:func:`pharmpy.modeling.set_first_order_elimination`
* :py:func:`pharmpy.modeling.set_michaelis_menten_elimination`
* :py:func:`pharmpy.modeling.set_mixed_mm_fo_elimination`
* :py:func:`pharmpy.modeling.set_zero_order_elimination`

Parameter variability model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pharmpy has multiple functions to change the parameter variability model. Using the
:ref:`base model<basic_models>` from above, we'll go through different aspects of changing the parameter variability
model.

Adding and removing parameter variability
=========================================

It is possible to add and remove inter-individual variability (IIV) and inter-occasion variability (IOV) using
:py:func:`pharmpy.modeling.add_iiv` and :py:func:`pharmpy.modeling.add_iov`. Since the start model from
:ref:`above<basic_models>` has IIV on all its parameters, we will start by removing an IIV using the
:py:func:`pharmpy.modeling.remove_iiv` function.

.. pharmpy-execute::
    run1 = remove_iiv(model_start, 'MAT')
    run1.random_variables.iiv

Next, we add an IIV to the same parameter:

.. pharmpy-execute::
    run2 = add_iiv(run1, 'MAT', 'exp', operation='*')
    run2.random_variables.iiv

And examine the model code:

.. pharmpy-execute::
   print_model_code(run2)

List of functions to add and remove parameter variability:

* :py:func:`pharmpy.modeling.add_iiv`
* :py:func:`pharmpy.modeling.add_iov`
* :py:func:`pharmpy.modeling.add_pk_iiv`
* :py:func:`pharmpy.modeling.remove_iiv`
* :py:func:`pharmpy.modeling.remove_iov`

Adding and removing covariance
==============================

As the next example, we will create a joint distribution using :py:func:`pharmpy.modeling.create_joint_distribution`
where the eta on MAT is included:

.. pharmpy-execute::
    run3 = create_joint_distribution(run2)
    run3.random_variables.iiv

And examine the model code:

.. pharmpy-execute::
   print_model_code(run3)

List of functions to change covariance structure:

* :py:func:`pharmpy.modeling.create_joint_distribution`
* :py:func:`pharmpy.modeling.split_joint_distribution`

Eta transformations
===================

It is also possible to transform the etas using the following functions:

* :py:func:`pharmpy.modeling.transform_etas_boxcox`
* :py:func:`pharmpy.modeling.transform_etas_john_draper`
* :py:func:`pharmpy.modeling.transform_etas_tdist`

Covariates and allometry
~~~~~~~~~~~~~~~~~~~~~~~~

Covariate effects may be applied to a model using the :py:func:`pharmpy.modeling.add_covariate_effect`.

.. pharmpy-execute::
   :hide-output:

    run1 = add_covariate_effect(model_start, 'CL', 'WGT', 'pow', operation='*')

Here, *CL* indicates the name of the parameter onto which you want to apply the effect, *WGT* is the name of the
covariate, and *pow* (power function) is the effect you want to apply. The effect can be either
added or multiplied to the parameter, denoted by '*' or '+' (multiplied is default). We can examine the model code:

.. pharmpy-execute::

   print_model_code(run1)

Pharmpy also supports user formatted covariate effects.

.. pharmpy-execute::
   :hide-output:

   user_effect = '((cov/std) - median) * theta'
   run2 = add_covariate_effect(model_start, 'CL', 'WGT', user_effect, operation='*')

The covariate is denoted as *cov*, the theta as *theta* (or, if multiple thetas: *theta1*, *theta2* etc.), and the mean,
median, and standard deviation as *mean*, *median*, and *std* respectively. This is in order for
the names to be substituted with the correct symbols.

.. pharmpy-execute::

   print_model_code(run2)

List of functions for covariates and allometry:

* :py:func:`pharmpy.modeling.add_allometry`
* :py:func:`pharmpy.modeling.add_covariate_effect`
* :py:func:`pharmpy.modeling.remove_covariate_effect`

Population parameters
~~~~~~~~~~~~~~~~~~~~~

There are several functions to simplify changing population parameters, such as functions to change initial estimates
and fixing parameters. As a first example, let us fix some parameters with :py:func:`pharmpy.modeling.fix_parameters`:

.. pharmpy-execute::
    run1 = fix_parameters(model_start, ['POP_CL', 'POP_VC'])
    run1.parameters

Another function that may be useful would be setting the initial estimates with
:py:func:`pharmpy.modeling.set_initial_estimates`.

.. pharmpy-execute::
    run2 = set_initial_estimates(run1, {'IIV_CL': 0.05, 'IIV_VC': 0.05})
    run2.parameters

And then the final model code:

.. pharmpy-execute::

   print_model_code(run2)

List of functions to change population parameters:

* :py:func:`pharmpy.modeling.add_population_parameter`
* :py:func:`pharmpy.modeling.fix_or_unfix_parameters`
* :py:func:`pharmpy.modeling.fix_parameters`
* :py:func:`pharmpy.modeling.fix_parameters_to`
* :py:func:`pharmpy.modeling.set_initial_estimates`
* :py:func:`pharmpy.modeling.set_lower_bounds`
* :py:func:`pharmpy.modeling.set_upper_bounds`
* :py:func:`pharmpy.modeling.unconstrain_parameters`
* :py:func:`pharmpy.modeling.unfix_parameters`
* :py:func:`pharmpy.modeling.unfix_parameters_to`
* :py:func:`pharmpy.modeling.update_inits`

Error model
~~~~~~~~~~~~

Pharmpy supports several error models. As an example, let us set the error model to a combined error model (start
model had proportional error model) using :py:func:`pharmpy.modeling.set_combined_error_model`:

.. pharmpy-execute::
    run1 = set_combined_error_model(model_start)
    run1.statements.error

List of functions to change the error model:

* :py:func:`pharmpy.modeling.remove_error_model`
* :py:func:`pharmpy.modeling.set_additive_error_model`
* :py:func:`pharmpy.modeling.set_combined_error_model`
* :py:func:`pharmpy.modeling.set_dtbs_error_model`
* :py:func:`pharmpy.modeling.set_iiv_on_ruv`
* :py:func:`pharmpy.modeling.set_power_on_ruv`
* :py:func:`pharmpy.modeling.set_proportional_error_model`
* :py:func:`pharmpy.modeling.set_time_varying_error_model`
* :py:func:`pharmpy.modeling.set_weighted_error_model`
* :py:func:`pharmpy.modeling.use_thetas_for_error_stdev`

BLQ transformations
~~~~~~~~~~~~~~~~~~~

It is also possible to perform BLQ transformations using the :py:func:`pharmpy.modeling.transform_blq` function. If
using the M3 or M4 method the standard deviation statement is derived symbolically.

.. pharmpy-execute::
    run1 = transform_blq(model_start, method='m4', lloq=0.1)
    run1.statements.error

And examine the model code:

.. pharmpy-execute::

   print_model_code(run1)

List of functions to perform BLQ transformations:

* :py:func:`pharmpy.modeling.transform_blq`

Estimation steps
~~~~~~~~~~~~~~~~

Pharmpy can change the estimation steps. As an example, let us change the estimation method from FOCE to IMP and set
how many iterations to output (the ``PRINT`` option in NONMEM) using the :py:func:`pharmpy.modeling.set_estimation_step`
function:

.. pharmpy-execute::
    run1 = set_estimation_step(model_start, method='imp', keep_every_nth_iter=10)
    run1.estimation_steps

If we then examine the model code:

.. pharmpy-execute::

   print_model_code(run1)

List of functions to change the estimation steps:

* :py:func:`pharmpy.modeling.add_covariance_step`
* :py:func:`pharmpy.modeling.add_estimation_step`
* :py:func:`pharmpy.modeling.append_estimation_step_options`
* :py:func:`pharmpy.modeling.remove_covariance_step`
* :py:func:`pharmpy.modeling.remove_estimation_step`
* :py:func:`pharmpy.modeling.set_estimation_step`
* :py:func:`pharmpy.modeling.set_evaluation_step`

.. _modeling_dataset:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Examining and modifying dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Pharmpy dataset can be examined and modified with several help functions in Pharmpy. To read more about the
dataset, see the :ref:`dataset documentation<dataset>`.

List of dataset functions:

* :py:func:`pharmpy.modeling.add_time_after_dose`
* :py:func:`pharmpy.modeling.check_dataset`
* :py:func:`pharmpy.modeling.deidentify_data`
* :py:func:`pharmpy.modeling.drop_columns`
* :py:func:`pharmpy.modeling.drop_dropped_columns`
* :py:func:`pharmpy.modeling.expand_additional_doses`
* :py:func:`pharmpy.modeling.get_baselines`
* :py:func:`pharmpy.modeling.get_cmt`
* :py:func:`pharmpy.modeling.get_concentration_parameters_from_data`
* :py:func:`pharmpy.modeling.get_covariate_baselines`
* :py:func:`pharmpy.modeling.get_doseid`
* :py:func:`pharmpy.modeling.get_doses`
* :py:func:`pharmpy.modeling.get_evid`
* :py:func:`pharmpy.modeling.get_ids`
* :py:func:`pharmpy.modeling.get_mdv`
* :py:func:`pharmpy.modeling.get_number_of_individuals`
* :py:func:`pharmpy.modeling.get_number_of_observations`
* :py:func:`pharmpy.modeling.get_number_of_observations_per_individual`
* :py:func:`pharmpy.modeling.get_observations`
* :py:func:`pharmpy.modeling.list_time_varying_covariates`
* :py:func:`pharmpy.modeling.read_dataset_from_datainfo`
* :py:func:`pharmpy.modeling.remove_loq_data`
* :py:func:`pharmpy.modeling.set_covariates`
* :py:func:`pharmpy.modeling.set_dvid`
* :py:func:`pharmpy.modeling.translate_nmtran_time`
* :py:func:`pharmpy.modeling.undrop_columns`

Subjects
~~~~~~~~

An array of all subject IDs can be retrieved.

.. pharmpy-execute::

    model = read_model(path / "pheno_real.mod")
    get_ids(model)

The number of subjects in the dataset could optionally be retrieved directly.

.. pharmpy-execute::

    get_number_of_individuals(model)

Observations
~~~~~~~~~~~~

The observations of the dataset indexed on subject ID and the independent variable can be extracted.

.. pharmpy-execute::

    get_observations(model)

The total number of observations can optionally be retrieved directly.

.. pharmpy-execute::

    get_number_of_observations(model)

Dosing
~~~~~~

Extract dosing information
==========================

The doses of the dataset indexed on subject ID and the independent variable can be extracted.

.. pharmpy-execute::

    doses = get_doses(model)
    doses

All unique doses can be listed

.. pharmpy-execute::

    doses.unique()

as well as the largest and the smallest dose

.. pharmpy-execute::

    doses.min()

.. pharmpy-execute::

    doses.max()

Dose grouping
=============

It is possible to create a DOSEID that groups each dose period starting from 1.

.. pharmpy-execute::

    ser = get_doseid(model)
    ser

Time after dose
===============

Add a column for time after dose (TAD)

.. pharmpy-execute::

    model = add_time_after_dose(model)
    model.dataset['TAD']

Concentration parameters
========================

Extract pharmacokinetic concentration parameters from the dataset

.. pharmpy-execute::

    get_concentration_parameters_from_data(model)
