.. _amd_tmdd:

==========
AMD - TMDD
==========

Will develop the best TMDD model based on an input model or dataset.

~~~~~~~
Running
~~~~~~~

The code to initiate the AMD tool for a tmdd model:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools read_modelfit_results, run_structsearch

    start_model = read_model('path/to/model')

    res = run_amd(
                modeltype='tmdd',
                input=start_model,
                cl_init=2.0,
                vc_init=5.0,
                mat_init=3.0,
                search_space='PERIPHERALS([1,2]);ELIMINATION([FO,ZO])',
                dv_types={'drug': 1, 'target': 2, 'complex': 3}
    )

Arguments
~~~~~~~~~

.. _amd_tmdd_args:

The AMD arguments used for TMDD models can be seen below. Some are mandatory for this type of model
building while others are optional, and some AMD arguments are not used for this model type. If any
of the mandatory arguments is missing, the program till raise an error.

.. note::
   The name of the DVID column must be "DVID".
   

Mandatory
---------

+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| Argument                                          | Description                                                                                                     |
+===================================================+=================================================================================================================+
| ``input``                                         | Path to a dataset or start model object. See :ref:`input in amd<input_amd>`                                     |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``modeltype``                                     | Set to 'tmdd' for this model type.                                                                              |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``cl_init``                                       | Initial estimate for the population clearance                                                                   |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``vc_init``                                       | Initial estimate for the central compartment population volume                                                  |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``mat_init``                                      | Initial estimate for the mean absorption time (only for oral models)                                            |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+

Optional
--------

+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| Argument                                          | Description                                                                                                     |
+===================================================+=================================================================================================================+
| ``dv_types``                                      | Dictionary of DV types for multiple DVs (e.g. dv_types = {'target': 2}).                                        |
|                                                   | Allowed keys are: 'drug', 'target', 'complex', 'drug_tot' and 'target_tot'. (For TMDD models only)              |
|                                                   | For more information see :ref:`here<dv_types>`.                                                                 |
|                                                   | Default is None which means that all observations are treated as drug observations.                             |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+

.. note::
    In addition to these arguments, :ref:`common arguments<amd_args_common>` can be added.

~~~~~~~~~~~~~~~~~~~
Strategy components
~~~~~~~~~~~~~~~~~~~

For a description about the different model building strategies in AMD, see :ref:`Strategy<strategy_amd>`.
This section will cover the aspects that are specific to TMDD models.

Structural
~~~~~~~~~~

.. graphviz::

    digraph BST {
            node [fontname="Arial",shape="rect"];
            rankdir="LR";
            base [label="Input", shape="oval"]
            s0 [label="add structural covariates"]
            s1 [label="modelsearch"]
            s2 [label="structsearch"]

            base -> s0
            s0 -> s1
            s1 -> s2
        }


**Structural covariates**

The structural covariates are added directly to the starting model. If these cannot be added here (due to missing 
parameters for instance) they will be added at the start of the next covsearch run. Note that all structural
covariates are added all at once without any test or search.

If no structural covariates are specified, no default is used.

**Modelsearch**

In this step the best structural PK model is found.
The settings that the AMD tool uses for the modelsearch subtool can be seen in the table below.

+-------------------+----------------------------------------------------------------------------------------------------+
| Argument          | Setting                                                                                            |
+===================+====================================================================================================+
| ``search_space``  | ``'search_space'`` (As defined in :ref:`AMD options<amd_args_common>`)                             |
+-------------------+----------------------------------------------------------------------------------------------------+
| ``algorithm``     | 'reduced_stepwise'                                                                                 |
+-------------------+----------------------------------------------------------------------------------------------------+
| ``iiv_strategy``  | 'absorption_delay'                                                                                 |
+-------------------+----------------------------------------------------------------------------------------------------+
| ``rank_type``     | 'bic' (type: mixed)                                                                                |
+-------------------+----------------------------------------------------------------------------------------------------+
| ``cutoff``        | None                                                                                               |
+-------------------+----------------------------------------------------------------------------------------------------+

If no search space is given by the user, the default search space is dependent on the ``administration`` argument.

.. tabs::

   .. tab:: TMDD ORAL

      .. code-block::

          ABSORPTION([FO,ZO,SEQ-ZO-FO])
          ELIMINATION([MM, MIX-FO-MM])
          LAGTIME([OFF,ON])
          TRANSITS([0,1,3,10],*)
          PERIPHERALS([0,1])

   .. tab:: TMDD IV

      .. code-block::

          ELIMINATION(FO)
          PERIPHERALS([0,1,2])

   .. tab:: TMDD IV+ORAL

      .. code-block::

          ABSORPTION([FO,ZO,SEQ-ZO-FO])
          ELIMINATION([MM, MIX-FO-MM])
          LAGTIME([OFF,ON])
          TRANSITS([0,1,3,10],*)
          PERIPHERALS([0,1,2])
    

.. note::
   Before modelsearch is run, the dataset of the model is filtered so that it only contains PK data (i.e. DVIDs smaller than 2).
   Before running structsearch the dataset is replaced again with the original dataset containing all DVIDs.

**Structsearch**

The input model to the structsearch tool is the highest ranking (PK) model from modelsearch that has mixed-mm-fo elimination
(note that this model might not be the highest ranking overall). If no such model exists then the final model from modelsearch
will be used regardless of the elimination type.
The dataset of the input model is replaced with the original dataset containing all DVIDs.

The extra model is the highest-ranking model that has the same structural features as the input model but with one less
peripheral compartment. If no such model exists the extra model is set to ``None``.

For a TMDD model, structsearch is run to determine the best structural model. All input arguments are specified by
the user when initializing AMD.

+------------------------+----------------------------------------------------------------------------------------------------+
| Argument               | Setting                                                                                            |
+========================+====================================================================================================+
| ``modeltype``          | 'tmdd'                                                                                             |
+------------------------+----------------------------------------------------------------------------------------------------+
| ``dv_types``           | ``'dv_types'`` (As defined in :ref:`AMD input<amd_tmdd_args>`)                                     |
+------------------------+----------------------------------------------------------------------------------------------------+
| ``strictness``         | ``strictness`` (As defined in :ref:`AMD input<amd_tmdd_args>`)                                     |
+------------------------+----------------------------------------------------------------------------------------------------+
| ``extra_model``        | The same model as the inputted model with one less peripheral compartment, if such a model exists  |
|                        | in the modelsearch results and passed the strictness criteria. Otherwise None.                     |
+------------------------+----------------------------------------------------------------------------------------------------+
| ``extra_model_reults`` | The connected modelfit results object for the extra model, if any. Otherwise None.                 |
+------------------------+----------------------------------------------------------------------------------------------------+

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
| ``rank_type``     | 'mbic' (type: iiv)        |  'mbic' (type: iiv)                                                    |
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
| ``p_value``       | 0.05                                                                                               |
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
                modeltype="tmdd",
                administration="iv",
                cl_init=2.0,
                vc_init=5.0
    )

Model input and search space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specifying input model and search space:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools read_modelfit_results, run_structsearch

    start_model = read_model('path/to/model')

    res = run_amd(
                modeltype='tmdd',
                input=start_model,
                search_space='PERIPHERALS([1,2]);ELIMINATION([FO,ZO])',
                dv_types={'drug': 1, 'target': 2, 'complex': 3}
                cl_init=2.0,
                vc_init=5.0,
                mat_init=3.0
    )
