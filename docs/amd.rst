.. _amd:

=================================
Automatic Model Development (AMD)
=================================

The AMD tool is a general tool for fully automatic model development to decide the best model given either a dataset
or a starting model. The tool is a combination of the following tools: :ref:`modelsearch`, :ref:`structsearch`, :ref:`iivsearch`,
:ref:`iovsearch`, :ref:`ruvsearch`, :ref:`allometry`, and :ref:`covsearch`.

~~~~~~~
Running
~~~~~~~

The AMD tool is available both in Pharmpy/pharmr.

To initiate AMD in Python/R:

.. pharmpy-code::

    from pharmpy.tools import run_amd

    dataset_path = 'path/to/dataset'
    strategy = 'all'
    res = run_amd(input=dataset_path,
                  modeltype='basic_pk',
                  administration='oral',
                  strategy=strategy,
                  search_space='LET(CATEGORICAL, [SEX]); LET(CONTINUOUS, [AGE])',
                  allometric_variable='WGT',
                  occasion='VISI')

This will take a dataset as ``input``, where the ``modeltype`` has been specified to be a PK model and ``administration`` is oral. AMD will search
for the best structural model, IIV structure, and residual model in the order specified by ``strategy`` (see :ref:`strategy<strategy_amd>`). We specify the column SEX
as a ``categorical`` covariate and AGE as a ``continuous`` covariate. Finally, we declare the WGT-column as our
``allometric_variable``, VISI as our ``occasion`` column.

~~~~~~~~~
Arguments
~~~~~~~~~

.. _amd_args:

+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| Argument                                          | Description                                                                                                     |
+===================================================+=================================================================================================================+
| ``input``                                         | Path to a dataset or start model object. See :ref:`input_amd`                                                   |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``results``                                       | ModelfitResults if input is a model                                                                             |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``modeltype``                                     | Type of model to build (e.g. 'basic_pk', 'pkpd', 'drug_metabolite' or 'tmdd')                                   |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``administration``                                | Route of administration. One of 'iv', 'oral' or 'ivoral'                                                        |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``cl_init``                                       | Initial estimate for the population clearance (default is 0.01)                                                 |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``vc_init``                                       | Initial estimate for the central compartment population volume (default is 1)                                   |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``mat_init``                                      | Initial estimate for the mean absorption time (only for oral models, default is 0.1)                            |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``b_init``                                        | Initial estimate for the baseline effect (only for pkpd models, default is 0.1)                                 |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``emax_init``                                     | Initial estimate for the Emax (only for pkpd models, default is 0.1)                                            |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``ec50_init``                                     | Initial estimate for the EC50 (only for pkpd models, default is 0.1)                                            |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``met_init``                                      | Initial estimate for the mean equilibration time (only for pkpd models, default is 0.1)                         |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``search_space``                                  | MFL for :ref:`search space<search_space_amd>` of structural and covariate models                                |
|                                                   | (default depends on ``modeltype``)                                                                              |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``lloq_limit``                                    | Lower limit of quantification.                                                                                  |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``lloq_method``                                   | Method to use for handling lower limit of quantification. See :py:func:`pharmpy.modeling.transform_blq`.        |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| :ref:`strategy<strategy_amd>`                     | Strategy defining run order of the different subtools valid arguments are 'all' (deafult) and 'retries'         |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``allometric_variable``                           | Variable to use for allometry (default is name of column described as body weight)                              |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``occasion``                                      | Name of occasion column                                                                                         |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``strictness``                                    | :ref:`Strictness<strictness>` criteria for model selection.                                                     |
|                                                   | Default is "minimization_successful or                                                                          |
|                                                   | (rounding_errors and sigdigs>= 0.1)"                                                                            |
|                                                   | If ``strictness`` is set to ``None`` no strictness                                                              |
|                                                   | criteria are applied                                                                                            |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``mechanistic_covariates``                        | List of covariates to run in a separate prioritezed covsearch run.                                              |
|                                                   | The effects are extracted from the given search space                                                           |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``retries_strategy``                              | Decide how to use the retries tool. Valid options are 'skip', 'all_final' or 'final'. Default is 'all_final'    |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``seed``                                          | A random number generator or seed to use for steps with random sampling.                                        |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``dv_types``                                      | Dictionary of DV types for multiple DVs (e.g. dv_types = {'target': 2}). Default is None.                       |
|                                                   | Allowed keys are: 'drug', 'target' and 'complex'. (For TMDD models only)                                        |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``parameter_uncertainty_method``                  | Parameter uncertainty method to use. Currently implemented methods are: 'SANDWICH', 'CPG' and 'OFIM'.           |
|                                                   | For more information about these methods see                                                                    |
|                                                   | :py:func:`here<pharmpy.model.EstimationStep.parameter_uncertainty_method>`.                                     |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``ignore_datainfo_fallback``                      | Decide wether or not to use connected datainfo object to infer information about the model. If True, all        |
|                                                   | information regarding the model must be given explicitly by the user, such as the allometric varible. If False, |
|                                                   | such information is extracted using the datainfo, in the absence of arguments given by the user. Default        |
|                                                   | is False.                                                                                                       |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+

.. _input_amd:

~~~~~
Input
~~~~~

The AMD tool can use both a dataset and a model as input. If the input is a dataset (with corresponding
:ref:`datainfo file<datainfo>`), Pharmpy will create a model with the following attributes:

* Structural: one compartment, first order absorption (if ``administration`` is ``'oral'``), first order elimination
* IIV: CL and VC with covariance (``'iv'``) or CL and VC with covariance and MAT (``'oral'``)
* Residual: proportional error model
* Estimation steps: FOCE with interaction

If the input is a model, the model needs to be a PK model.

When running the tool for modeltype 'ivoral' with a dataset as input, the dataset is required to have a CMT column with values 1 
(oral doses) and 2 (IV doses). This is required for the creation of the initial one-compartment model with first order absorption. 
In order to easily differentiate the two doses, an administration ID (ADMID) column will be added to the data as well. This will be 
used in order to differentiate the different doses from one another with respect to the applied error model. If a model is used as 
input instead, this is not applied as it is assumed to have the correct CMT values for the connected model, along with a way of 
differentiating the doses from one another.

.. warning::
    The AMD tool, or more specifically the :ref:`modelsearch` tool, does not support NONMEM models with a RATE
    column. This needs to be dropped (either via model or datainfo file) or excluded from the dataset.

.. _search_space_amd:

~~~~~~~~~~~~
Search space
~~~~~~~~~~~~

.. note::
    Please see the description of :ref:`mfl` for how to define the search space for the structural and covariate models.

The search space has different defaults depending on which type of data has been inputed. For a PK oral model, the
default is:

.. code-block::

    ABSORPTION([FO,ZO,SEQ-ZO-FO])
    ELIMINATION(FO)
    LAGTIME([OFF,ON])
    TRANSITS([0,1,3,10],*)
    PERIPHERALS(0,1)
    COVARIATE?(@IIV, @CONTINUOUS, *)
    COVARIATE?(@IIV, @CATEGORICAL, CAT)

For a PK IV model, the default is:

.. code-block::

    ELIMINATION(FO)
    PERIPHERALS([0,1,2])
    COVARIATE?(@IIV, @CONTINUOUS, *)
    COVARIATE?(@IIV, @CATEGORICAL, CAT)
    
For a PK IV+ORAL model, the default is:

.. code-block::

    ABSORPTION([FO,ZO,SEQ-ZO-FO])
    ELIMINATION(FO)
    LAGTIME([OFF,ON])
    TRANSITS([0,1,3,10],*)
    PERIPHERALS([0,1,2])
    COVARIATE?(@IIV, @CONTINUOUS, *)
    COVARIATE?(@IIV, @CATEGORICAL, CAT)

Note that defaults are overriden selectively: structural model features
defaults will be ignored as soon as one structural model feature is explicitly
given, but the covariate model defaults will stay in place, and vice versa. For
instance, if one defines ``search_space`` as ``LAGTIME(ON)``, the effective
search space will be as follows:

.. code-block::

    LAGTIME(ON)
    COVARIATE?(@IIV, @CONTINUOUS, *)
    COVARIATE?(@IIV, @CATEGORICAL, CAT)

.. _strategy_amd:

~~~~~~~~~~~~~~~~~~~~~~~~
Strategy for running AMD
~~~~~~~~~~~~~~~~~~~~~~~~

There are different strategies available for running the AMD tool which is specified
in the ``strategy`` argument. They all use a combination of the different subtools
described below and will be described below. As all tools might not be applicable to
all model types, the used subtools in the different steps is dependent on the
``modeltype`` argument. Please see the description for each tool described below
for more details.

all (default)
~~~~~~~~~~~~~

If no argument is specified, 'all' will be used as the default strategy. This will use
all tools available for the specified ``modeltype``. The exact workflow can hence differ for
the different model type but a general visualization of this can be seen below:

.. graphviz::

    digraph BST {
            node [fontname="Arial",shape="rect"];
            rankdir="LR";
            base [label="Input", shape="oval"]
            s0 [label="structural"]
            s1 [label="iivsearch"]
            s2 [label="residual"]
            s3 [label="iovsearch"]
            s4 [label="allometry"]
            s5 [label="covariates"]
            s6 [label="results", shape="oval"]

            base -> s0
            s0 -> s1
            s1 -> s2
            s2 -> s3
            s3 -> s4
            s4 -> s5
            s5 -> s6
        }


retries
~~~~~~~

The retries strategy is an extension of the 'all' strategy. It is defined by the re-running
of IIVsearch and RUVsearch. This indicate that the tool follow the exact same principles
and the workflow hence is dependent on the model type in question.

The general order of subtools hence become:

.. graphviz::

    digraph BST {
            node [fontname="Arial",shape="rect"];
            rankdir="LR";
            base [label="Input", shape="oval"]
            s0 [label="structural"]
            s1 [label="iivsearch"]
            s2 [label="residual"]
            s3 [label="iovsearch"]
            s4 [label="allometry"]
            s5 [label="covariates"]
            s6 [label="rerun_iivsearch"]
            s7 [label="rerun_ruvsearch"]
            s8 [label="results", shape="oval"]

            base -> s0
            s0 -> s1
            s1 -> s2
            s2 -> s3
            s3 -> s4
            s4 -> s5
            s5 -> s6
            s6 -> s7
            s7 -> s8
        }
        
SIR
~~~

This strategy is related to 'SRI' and 'RSI' and is an acronym for running
the Structural, IIVsearch and RUVsearch part of the AMD tool. The workflow hence
become as follows:

.. graphviz::

    digraph BST {
            node [fontname="Arial",shape="rect"];
            rankdir="LR";
            base [label="Input", shape="oval"]
            s0 [label="structural"]
            s1 [label="iivsearch"]
            s2 [label="residual"]
            s3 [label="results", shape="oval"]

            base -> s0
            s0 -> s1
            s1 -> s2
            s2 -> s3
        }

SRI
~~~

This strategy is related to 'SIR' and 'RSI' and is an acronym for running
the Structural, RUVsearch and IIVsearch part of the AMD tool. The workflow hence
become as follows:

.. graphviz::

    digraph BST {
            node [fontname="Arial",shape="rect"];
            rankdir="LR";
            base [label="Input", shape="oval"]
            s0 [label="structural"]
            s1 [label="residual"]
            s2 [label="iivsearch"]
            s3 [label="results", shape="oval"]

            base -> s0
            s0 -> s1
            s1 -> s2
            s2 -> s3
        }
        
RSI
~~~

This strategy is related to 'SIR' and 'SRI' and is an acronym for running
the RUVsearch, Structural and IIVsearch part of the AMD tool. The workflow hence
become as follows:

.. graphviz::

    digraph BST {
            node [fontname="Arial",shape="rect"];
            rankdir="LR";
            base [label="Input", shape="oval"]
            s0 [label="residual"]
            s1 [label="structural"]
            s2 [label="iivsearch"]
            s3 [label="results", shape="oval"]

            base -> s0
            s0 -> s1
            s1 -> s2
            s2 -> s3
        }

~~~~~~~~~~~~~~~~~~~~
Subtools used in AMD
~~~~~~~~~~~~~~~~~~~~

The default algorithms for six tools in amd can be seen in the table below. For more details regarding the settings
for each subtool, see the respective subheading.

+------------------+-----------------------------------------------------------------------------------------+------------------------------------+
| Tool             | Description                                                                             | Can be used with ``modetype``      |
+==================+=========================================================================================+====================================+
| modelsearch      | Search for best structural model for a PK model, includes absorption, distribution, and | ``basic_pk``, ``drug_metabolite``, |
|                  | elimination (part of 'structural' AMD step)                                             | ``tmdd``                           |
+------------------+-----------------------------------------------------------------------------------------+------------------------------------+
| structsearch     | Search for best structural model. Includes PKPD, TMDD and drug metabolite models        | ``pkpd``, ``drug_metabolite``,     |
|                  | (part of 'structural' AMD step)  ´                                                       | ``tmdd``                           |
+------------------+-----------------------------------------------------------------------------------------+------------------------------------+
| iivsearch        | Search for best IIV structure, both in terms of number of IIVs to keep as well as       | All model types                    |
|                  | covariance structure                                                                    |                                    |
+------------------+-----------------------------------------------------------------------------------------+------------------------------------+
| iovsearch        | Search for best IOV structure and remove IIVs explained by IOV                          | All model types                    |
+------------------+-----------------------------------------------------------------------------------------+------------------------------------+
| ruvsearch        | Search for best residual error model, test IIV on RUV, power on RUV, combined error     | All model types                    |
|                  | model, and time-varying                                                                 |                                    |
+------------------+-----------------------------------------------------------------------------------------+------------------------------------+
| allometry        | Test allometric scaling                                                                 | ``basic_pk``, ``drug_metabolite``, |
|                  |                                                                                         | ``tmdd``                           |
+------------------+-----------------------------------------------------------------------------------------+------------------------------------+
| covsearch        | Test and identify covariate effects                                                     | All model types                    |
+------------------+-----------------------------------------------------------------------------------------+------------------------------------+

Structural
~~~~~~~~~~

This subtool selects the best structural model, using the appropriate subtools for the chosen ``modeltype``. For regular PK
analysis, modelsearch will be used. For structural components connected to PD, metabolite or TMDD however, structsearch
will be used. See :ref:`modelsearch` or :ref:`structsearch` for more details about the tool.
In this stage, structural covariate effects are also added (all at once) to the starting model. Please see :ref:`covsearch` 
for more information of this.

If structural components are to be run, they will be run in the order below.

.. graphviz::

    digraph BST {
            node [fontname="Arial",shape="rect"];
            rankdir="LR";
            base [label="Input", shape="oval"]
            s0 [label="structural covariates"]
            s1 [label="modelsearch"]
            s2 [label="structsearch"]

            base -> s0
            s0 -> s1
            s1 -> s2
        }

Modelsearch
===========

The settings that the AMD tool uses for the modelsearch subtool can be seen in the table below.

+---------------+----------------------------------------------------------------------------------------------------+
| Argument      | Setting                                                                                            |
+===============+====================================================================================================+
| search_space  | Given in :ref:`AMD options<amd_args>` (``search_space``)                                           |
+---------------+----------------------------------------------------------------------------------------------------+
| algorithm     | ``'reduced_stepwise'``                                                                             |
+---------------+----------------------------------------------------------------------------------------------------+
| iiv_strategy  | ``'absorption_delay'``                                                                             |
+---------------+----------------------------------------------------------------------------------------------------+
| rank_type     | ``'bic'`` (type: mixed)                                                                            |
+---------------+----------------------------------------------------------------------------------------------------+
| cutoff        | ``None``                                                                                           |
+---------------+----------------------------------------------------------------------------------------------------+

Structsearch
============

The structsearch tool selects the best structural model from a set of models. Currently implemented
model types are PKPD, TMDD and drug-metabolite.

In order to run AMD for a pkpd model the ``modeltype`` needs to be set to `pkpd`. For running drug metabolite models, 
the expected ``modeltype`` needs to be set to `drug_metabolite`

.. note::
    Please note that it is only possible to run the AMD tool for the PD part of PKPD models. The tool
    expects a fully build PK model as input. 


IIVsearch
~~~~~~~~~

This subtool selects the IIV structure, see :ref:`iivsearch` for more details about the tool. The settings
that the AMD tool uses for this subtool can be seen in the table below.


+---------------+---------------------------+------------------------------------------------------------------------+
| Argument      | Setting                   |   Setting (rerun)                                                      |
+===============+===========================+========================================================================+
| algorithm     | ``'brute_force'``         |  ``'brute_force'``                                                     |
+---------------+---------------------------+------------------------------------------------------------------------+
| iiv_strategy  | ``'fullblock'``           |  ``'no_add'``                                                          |
+---------------+---------------------------+------------------------------------------------------------------------+
| rank_type     | ``'bic'`` (type: iiv)     |  ``'bic'`` (type: iiv)                                                 |
+---------------+---------------------------+------------------------------------------------------------------------+
| cutoff        | ``None``                  |  ``None``                                                              |
+---------------+---------------------------+------------------------------------------------------------------------+

IOVsearch
~~~~~~~~~

This subtool selects the IOV structure and tries to remove corresponding IIVs if possible, see :ref:`iovsearch` for
more details about the tool. The settings that the AMD tool uses for this subtool can be seen in the table below. If no
argument for ``occasion`` is given, this tool will not be run.

+---------------------+----------------------------------------------------------------------------------------------+
| Argument            | Setting                                                                                      |
+=====================+==============================================================================================+
| column              | Given in :ref:`AMD options<amd_args>` (``occasion``)                                         |
+---------------------+----------------------------------------------------------------------------------------------+
| list_of_parameters  | ``None``                                                                                     |
+---------------------+----------------------------------------------------------------------------------------------+
| rank_type           | ``'bic'`` (type: random)                                                                     |
+---------------------+----------------------------------------------------------------------------------------------+
| cutoff              | ``None``                                                                                     |
+---------------------+----------------------------------------------------------------------------------------------+
| distribution        | ``'same-as-iiv'``                                                                            |
+---------------------+----------------------------------------------------------------------------------------------+

Residual
~~~~~~~~

This subtool selects the residual model, see :ref:`ruvsearch` for more details about the tool. The settings
that the AMD tool uses for this subtool can be seen in the table below. When re-running the tool, the settings remain
the same.


+---------------+----------------------------------------------------------------------------------------------------+
| Argument      | Setting                                                                                            |
+===============+====================================================================================================+
| groups        | ``4``                                                                                              |
+---------------+----------------------------------------------------------------------------------------------------+
| p_value       | ``0.05``                                                                                           |
+---------------+----------------------------------------------------------------------------------------------------+
| skip          | ``None``                                                                                           |
+---------------+----------------------------------------------------------------------------------------------------+

Allometry
~~~~~~~~~

This subtool tries to apply allometry, see :ref:`allometry` for more details about the tool. The settings
that the AMD tool uses for this subtool can be seen in the table below. Please note that if ``ignore_datainfo_fallback`` is
set to ``True`` and no allometric variable is given, this tool will not be run. 

+----------------------+---------------------------------------------------------------------------------------------+
| Argument             | Setting                                                                                     |
+======================+=============================================================================================+
| allometric_variable  | Given in :ref:`AMD options<amd_args>` (``allometric_variable``)                             |
+----------------------+---------------------------------------------------------------------------------------------+
| reference_value      | ``70``                                                                                      |
+----------------------+---------------------------------------------------------------------------------------------+
| parameters           | ``None``                                                                                    |
+----------------------+---------------------------------------------------------------------------------------------+
| initials             | ``None``                                                                                    |
+----------------------+---------------------------------------------------------------------------------------------+
| lower_bounds         | ``None``                                                                                    |
+----------------------+---------------------------------------------------------------------------------------------+
| upper_bounds         | ``None``                                                                                    |
+----------------------+---------------------------------------------------------------------------------------------+
| fixed                | ``None``                                                                                    |
+----------------------+---------------------------------------------------------------------------------------------+

.. note::
    This tool is skipped if ``modeltype = 'pkpd'``

Covariates
~~~~~~~~~~

This subtool selects which covariate effects to apply, see :ref:`covsearch` for more details about the tool. The
settings that the AMD tool uses for this subtool can be seen in the table below. Please note that if ``ignore_datainfo_fallback``
is set to ``True`` and no covariates are given, this tool will not be run.

+---------------+----------------------------------------------------------------------------------------------------+
| Argument      | Setting                                                                                            |
+===============+====================================================================================================+
| effects       | Given in :ref:`AMD options<amd_args>` (``search_space``)                                           |
+---------------+----------------------------------------------------------------------------------------------------+
| p_forward     | ``0.05``                                                                                           |
+---------------+----------------------------------------------------------------------------------------------------+
| p_backward    | ``0.01``                                                                                           |
+---------------+----------------------------------------------------------------------------------------------------+
| max_steps     | ``-1``                                                                                             |
+---------------+----------------------------------------------------------------------------------------------------+
| algorithm     | ``'scm-forward-then-backward'``                                                                    |
+---------------+----------------------------------------------------------------------------------------------------+

For an entire AMD run, it is possible to get a maximum of three covsearch runs, which are described below:

+---------------------+-----------------------------------------------------------------------------------------+
| Type of covsearch   | Description                                                                             |
+=====================+=========================================================================================+
| Structural          | Performed in the structural part of the AMD run. The structural covariates are added    |
|                     | directly to the starting model.                                                         |
|                     | If these cannot be added here (due to missing parameters for instance) they will        |
|                     | be tested once more at the start of the next covsearch run.                             |
+---------------------+-----------------------------------------------------------------------------------------+
| Mechanistic         | If any mechanistic covariates have been given as input to the AMD tool, the specified   |
|                     | covariate effects for these covariates is run in a separate initial covsearch run When  |
|                     | adding covariates.                                                                      |
+---------------------+-----------------------------------------------------------------------------------------+
| Exploratory         | The remaining covariates are tested after all mechanistic covariates have been tested.  |
+---------------------+-----------------------------------------------------------------------------------------+

Retries
~~~~~~~~~~

If ``retries_strategy`` is set to 'all_final', the retries tool will be run on the final model from each subtool.
With the argument set to 'final', the retries tool will only be run on the final model from the last subtool.
Finally, if the argument is set to 'skip', no retries will be performed. See :ref:`retries` for more details about the 
tool. When running the tool from AMD, the settings below will be used.

If argument ``seed`` is set, the chosen seed or random number generator will be used for the random sampling within the
tool.

+----------------------+----------------------------------------------------------------------------------------------------+
| Argument             | Setting                                                                                            |
+======================+====================================================================================================+
| number_of_candidates | ``5``                                                                                              |
+----------------------+----------------------------------------------------------------------------------------------------+
| fraction             | ``0.1``                                                                                            |
+----------------------+----------------------------------------------------------------------------------------------------+
| scale                | ``UCP``                                                                                            |
+----------------------+----------------------------------------------------------------------------------------------------+
| use_initial_estimates| False                                                                                              |
+----------------------+----------------------------------------------------------------------------------------------------+
| prefix_name          | The name of the previously run tool                                                                |
+----------------------+----------------------------------------------------------------------------------------------------+

~~~~~~~
Results
~~~~~~~

The results object contains the final selected model and various summary tables, all of which can be accessed in the
results object as well as files in .csv/.json format.

The ``summary_tool`` table contains information such as which feature each model candidate has, the difference to the
start model (in this case comparing BIC), and final ranking:

.. pharmpy-execute::
    :hide-code:

    from pharmpy.workflows.results import read_results
    res = read_results('tests/testdata/results/amd_results.json')
    res.summary_tool

To see information about the actual model runs, such as minimization status, estimation time, and parameter estimates,
you can look at the ``summary_models`` table. The table is generated with
:py:func:`pharmpy.modeling.summarize_modelfit_results`.

.. pharmpy-execute::
    :hide-code:

    res.summary_models

Finally, you can see a summary of any errors and warnings of the final selected model in ``summary_errors``.
See :py:func:`pharmpy.tools.summarize_errors` for information on the content of this table.

.. pharmpy-execute::
    :hide-code:

    import pandas as pd
    pd.set_option('display.max_colwidth', None)
    res.summary_errors


Final model
~~~~~~~~~~~

Some plots and tables on the final model can be found both in the amd report and in the results object.

.. pharmpy-execute::
   :hide-code:

   res.final_model_parameter_estimates.style.format({
       'estimates': '{:,.4f}'.format,
       'RSE': '{:,.1%}'.format,
   })


.. pharmpy-execute::
   :hide-code:

   res.final_model_dv_vs_pred_plot


.. pharmpy-execute::
   :hide-code:

   res.final_model_dv_vs_ipred_plot


.. pharmpy-execute::
   :hide-code:

   res.final_model_abs_cwres_vs_ipred_plot


.. pharmpy-execute::
   :hide-code:

   res.final_model_cwres_vs_idv_plot
