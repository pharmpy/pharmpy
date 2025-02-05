.. _amd_pk:

========
AMD - PK
========

Will develop a PK model based on a model or dataset input.

~~~~~~~
Running
~~~~~~~

The code to initiate the AMD tool for a PK model:

.. pharmpy-code::

    from pharmpy.tools import run_amd

    res = run_amd(input='path/to/dataset',
                  modeltype='basic_pk',
                  administration='oral',
                  cl_init=2.0,
                  vc_init=5.0,
                  mat_init=3.0,
                  strategy='default',
                  search_space='ABSORPTION(FO);ELIMINATION(ZO)',
                  allometric_variable='WGT',
                  occasion='VISI'
    )

Arguments
~~~~~~~~~

.. _amd_pk_args:

The arguments used in PK models can be seen below. Some are mandatory for this type of model
building while others are optional, and some AMD arguments are not used for this model type.
If any of the mandatory arguments are missing, the model will not be run. The only exception
being for ``mat_init`` which is only mandatory for 'oral' administration.

Mandatory
---------

+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| Argument                                          | Description                                                                                                     |
+===================================================+=================================================================================================================+
| ``input``                                         | Path to a dataset or start model object. See :ref:`input in amd<input_amd>`                                     |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``modeltype``                                     | Set to 'basic_pk' for this model type.                                                                          |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``cl_init``                                       | Initial estimate for the population clearance                                                                   |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``vc_init``                                       | Initial estimate for the central compartment population volume                                                  |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``mat_init``                                      | Initial estimate for the mean absorption time (only for oral models)                                            |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+

.. note::
    In addition to these arguments, :ref:`common arguments<amd_args_common>` can be added.

~~~~~~~~~~~~~~~~~~~
Strategy components
~~~~~~~~~~~~~~~~~~~

For a description about the different model building strategies in AMD, see :ref:`Strategy<strategy_amd>`.
This section will cover the aspects that are specific to PK models.

Structural
~~~~~~~~~~

.. graphviz::

    digraph BST {
            node [fontname="Arial",shape="rect"];
            rankdir="LR";
            base [label="Input", shape="oval"]
            s0 [label="add structural covariates"]
            s1 [label="modelsearch"]

            base -> s0
            s0 -> s1
        }


**Structural covariates**

The structural covariates are added directly to the starting model. If these cannot be added here (due to missing 
parameters for instance) they will be added at the start of the next covsearch run. Note that all structural
covariates are added all at once without any test or search.

If no structural covariates are specified, no default is used.

**Modelsearch**

The settings that the AMD tool uses for the modelsearch subtool can be seen in the table below.

+-------------------+----------------------------------------------------------------------------------------------------+
| Argument          | Setting                                                                                            |
+===================+====================================================================================================+
| ``search_space``  | ``search_space`` (As defined in :ref:`AMD options<amd_args_common>`)                               |
+-------------------+----------------------------------------------------------------------------------------------------+
| ``algorithm``     | 'reduced_stepwise'                                                                                 |
+-------------------+----------------------------------------------------------------------------------------------------+
| ``iiv_strategy``  | 'absorption_delay'                                                                                 |
+-------------------+----------------------------------------------------------------------------------------------------+
| ``rank_type``     | 'bic' (type: mixed)                                                                                |
+-------------------+----------------------------------------------------------------------------------------------------+
| ``cutoff``        | None                                                                                               |
+-------------------+----------------------------------------------------------------------------------------------------+

If no search space is given by the user, the default search space is dependent on the ``administration`` argument

.. tabs::

   .. tab:: PK ORAL

      .. code-block::

          ABSORPTION([FO,ZO,SEQ-ZO-FO])
          ELIMINATION(FO)
          LAGTIME([OFF,ON])
          TRANSITS([0,1,3,10],*)
          PERIPHERALS([0,1])

   .. tab:: PK IV

      .. code-block::

          ELIMINATION(FO)
          PERIPHERALS([0,1,2])

   .. tab:: PK IV+ORAL

      .. code-block::

          ABSORPTION([FO,ZO,SEQ-ZO-FO])
          ELIMINATION(FO)
          LAGTIME([OFF,ON])
          TRANSITS([0,1,3,10],*)
          PERIPHERALS([0,1,2])

IIVSearch
~~~~~~~~~

The settings that the AMD tool uses for this subtool can be seen in the table below.

+-------------------+---------------------------+------------------------------------------------------------------------+
| Argument          | Setting                   |   Setting (rerun)                                                      |
+===================+===========================+========================================================================+
| ``algorithm``     | 'top_down_exhaustive'     |  'top_down_exhaustive'                                                 |
+-------------------+---------------------------+------------------------------------------------------------------------+
| ``iiv_strategy``  | 'fullblock'               |  'no_add'                                                              |
+-------------------+---------------------------+------------------------------------------------------------------------+
| ``rank_type``     | 'bic' (type: iiv)         |  'bic' (type: iiv)                                                     |
+-------------------+---------------------------+------------------------------------------------------------------------+
| ``cutoff``        | None                      |  None                                                                  |
+-------------------+---------------------------+------------------------------------------------------------------------+
| ``keep``          | Clearance parameters      | Clearance parameters from input model                                  |
|                   | from input model          |                                                                        |
+-------------------+---------------------------+------------------------------------------------------------------------+

Residual
~~~~~~~~

The settings that the AMD tool uses for this subtool can be seen in the table below. When re-running the tool, the
settings remain the same.

+-------------------+----------------------------------------------------------------------------------------------------+
| Argument          | Setting                                                                                            |
+===================+====================================================================================================+
| ``groups``        | 4                                                                                                  |
+-------------------+----------------------------------------------------------------------------------------------------+
| ``p_value``       | 0.001                                                                                              |
+-------------------+----------------------------------------------------------------------------------------------------+
| ``skip``          | None                                                                                               |
+-------------------+----------------------------------------------------------------------------------------------------+

IOVSearch
~~~~~~~~~

The settings that the AMD tool uses for this subtool can be seen in the table below. 

+-------------------------+----------------------------------------------------------------------------------------------+
| Argument                | Setting                                                                                      |
+=========================+==============================================================================================+
| ``column``              | ``occasion`` (As defined in :ref:`AMD options<amd_args_common>`)                             |
+-------------------------+----------------------------------------------------------------------------------------------+
| ``list_of_parameters``  | None                                                                                         |
+-------------------------+----------------------------------------------------------------------------------------------+
| ``rank_type``           | 'bic' (type: random)                                                                         |
+-------------------------+----------------------------------------------------------------------------------------------+
| ``cutoff``              | None                                                                                         |
+-------------------------+----------------------------------------------------------------------------------------------+
| ``distribution``        | 'same-as-iiv'                                                                                |
+-------------------------+----------------------------------------------------------------------------------------------+

Allometry
~~~~~~~~~

The settings that the AMD tool uses for this subtool can be seen in the table below.

+--------------------------+---------------------------------------------------------------------------------------------+
| Argument                 | Setting                                                                                     |
+==========================+=============================================================================================+
| ``allometric_variable``  | ``allometric_variable`` (As defined in :ref:`AMD options<amd_args_common>`)                 |
+--------------------------+---------------------------------------------------------------------------------------------+
| ``reference_value``      | 70                                                                                          |
+--------------------------+---------------------------------------------------------------------------------------------+
| ``parameters``           | None                                                                                        |
+--------------------------+---------------------------------------------------------------------------------------------+
| ``initials``             | None                                                                                        |
+--------------------------+---------------------------------------------------------------------------------------------+
| ``lower_bounds``         | None                                                                                        |
+--------------------------+---------------------------------------------------------------------------------------------+
| ``upper_bounds``         | None                                                                                        |
+--------------------------+---------------------------------------------------------------------------------------------+
| ``fixed``                | None                                                                                        |
+--------------------------+---------------------------------------------------------------------------------------------+

covsearch
~~~~~~~~~

.. graphviz::

    digraph BST {
            node [fontname="Arial",shape="rect"];
            rankdir="LR";
            base [label="Input", shape="oval"]
            s0 [label="mechanistic covariates"]
            s1 [label="exploratory covariates"]

            base -> s0
            s0 -> s1
        }

The settings that the AMD tool uses for this subtool can be seen in the table below.

+-------------------+----------------------------------------------------------------------------------------------------+
| Argument          | Setting                                                                                            |
+===================+====================================================================================================+
| ``search_space``  | ``search_space`` (As defined in :ref:`AMD options<amd_args_common>`)                               |
+-------------------+----------------------------------------------------------------------------------------------------+
| ``p_forward``     | 0.05                                                                                               |
+-------------------+----------------------------------------------------------------------------------------------------+
| ``p_backward``    | 0.01                                                                                               |
+-------------------+----------------------------------------------------------------------------------------------------+
| ``max_steps``     | -1                                                                                                 |
+-------------------+----------------------------------------------------------------------------------------------------+
| ``algorithm``     | 'scm-forward-then-backward'                                                                        |
+-------------------+----------------------------------------------------------------------------------------------------+

If no search space for this tool is given, the following default will be used:

.. code-block::

    COVARIATE?(@IIV, @CONTINUOUS, exp, *)
    COVARIATE?(@IIV, @CATEGORICAL, cat, *)

Here, both statements are defined with a '?', meaning that these are covariate effect(s) to be explored rather than
structural covariate effects, which are added during the earlier "structural" step.

**Mechanisitic covariates**

If any mechanistic covariates have been given as input to the AMD tool, the specified covariate effects for these
covariates is run in a separate initial covsearch run when adding covariates. These covariate effects are extracted
from the given search space

**Exploratory covariates**

The remaining covariate effects from the search space are now run in an exploratory search.

~~~~~~~~
Examples
~~~~~~~~

Minimal
~~~~~~~

A minimal example for running AMD with model type PK:

.. pharmpy-code::

    from pharmpy.tools import run_amd

    dataset_path = 'path/to/dataset'

    res = run_amd(
                dataset_path,
                modeltype="basic_pk",
                administration="iv",
                cl_init=2.0,
                vc_init=5.0
    )

Model input and search space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specifying input model and search space:

.. pharmpy-code::

    from pharmpy.tools import run_amd

    start_model = read_model('path/to/model')

    res = run_amd(
                input=start_model,
                modeltype='basic_pk',
                administration='oral'
                search_space='ABSORPTION(FO);ELIMINATION([FO,ZO]);COVARIATE(CL, WGT, POW)',
                cl_init=2.0,
                vc_init=5.0,
    )
