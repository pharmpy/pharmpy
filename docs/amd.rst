.. _amd:

=================================
Automatic Model Development (AMD)
=================================

The AMD tool is a general tool for fully automatic model development to decide the best model given either a dataset
or a starting model. The tool is a combination of the following tools: :ref:`modelsearch`, :ref:`structsearch`, :ref:`iivsearch`,
:ref:`iovsearch`, :ref:`ruvsearch`, :ref:`allometry`, and :ref:`covsearch`.

On this page, general information regarding the AMD workflow can be found.

~~~~~~~~~~~~~~~~~~~~~
Supported model types
~~~~~~~~~~~~~~~~~~~~~

.. _modeltypes_amd:

AMD currently supports a few different model types. For specific information regarding a given model type,
please see their respective page.

.. toctree::
   :maxdepth: 1

   PK <amd_pk>
   PKPD <amd_pkpd>
   TMDD <amd_tmdd>
   Drug Metabolite <amd_drug_metabolite>


~~~~~~~
Running
~~~~~~~

The AMD tool is available both in Pharmpy/pharmr.

To initiate AMD in Python/R:

.. pharmpy-code::

    from pharmpy.tools import run_amd

    search_space = 'COVARIATE?(@IIV,SEX,EXP);COVARIATE?(@IIV,AGE,CAT)'
    res = run_amd(input='path/to/dataset',
                  modeltype='basic_pk',
                  administration='oral',
                  cl_init=2.0,
                  vc_init=5.0,
                  mat_init=3.0,
                  strategy='default',
                  search_space=search_space,
                  allometric_variable='WGT',
                  occasion='VISI'
    )

This will take a dataset as ``input``, where the ``modeltype`` has been specified to be a PK model and ``administration`` is oral. AMD will search
for the best structural model, IIV structure, and residual error model in the order specified by ``strategy`` (see :ref:`strategy<strategy_amd>`). We specify the column SEX
as a ``categorical`` covariate and AGE as a ``continuous`` covariate. Finally, we declare the WGT-column as our
``allometric_variable``, VISI as our ``occasion`` column.

~~~~~~~~~
Arguments
~~~~~~~~~

.. _amd_args:

Model type specific arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| Argument                                          | Description                                                                                                     |
+===================================================+=================================================================================================================+
| ``input``                                         | Path to a dataset or start model object. See :ref:`input_amd`                                                   |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``results``                                       | ModelfitResults if input is a model                                                                             |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``modeltype``                                     | Type of model to build (e.g. 'basic_pk', 'pkpd', 'drug_metabolite' or 'tmdd'). Default is 'basic_pk'            |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``administration``                                | Route of administration. One of 'iv', 'oral' or 'ivoral'. Default is 'oral'                                     |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``cl_init``                                       | Initial estimate for the population clearance                                                                   |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``vc_init``                                       | Initial estimate for the central compartment population volume                                                  |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``mat_init``                                      | Initial estimate for the mean absorption time (only for oral models)                                            |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``b_init``                                        | Initial estimate for the baseline effect (only for PKPD models)                                                 |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``emax_init``                                     | Initial estimate for the Emax (only for PKPD models)                                                            |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``ec50_init``                                     | Initial estimate for the EC50 (only for PKPD models)                                                            |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``met_init``                                      | Initial estimate for the mean equilibration time (only for PKPD models)                                         |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``dv_types``                                      | Dictionary of DV types for multiple DVs (e.g. ``dv_types = {'target': 2})``. Default is None.                   |
|                                                   | (For TMDD models only)                                                                                          |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+


General arguments
~~~~~~~~~~~~~~~~~

.. _amd_args_common:

+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| Argument                                          | Description                                                                                                     |
+===================================================+=================================================================================================================+
| ``strategy``                                      | :ref:`Strategy<strategy_amd>` defining run order of the different subtools. Default is 'default'                |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``search_space``                                  | MFL for :ref:`search space<search_space_amd>` of structural and covariate models                                |
|                                                   | (default depends on ``modeltype`` and ``administration``)                                                       |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``lloq_limit``                                    | Lower limit of quantification.                                                                                  |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``lloq_method``                                   | Method to use for handling lower limit of quantification. See :py:func:`pharmpy.modeling.transform_blq`.        |
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
| ``mechanistic_covariates``                        | List of covariates or covariate/parameter combinations to run in a separate prioritized covsearch run. Allowed  |
|                                                   | elements in the list are strings of covariates or tuples with one covariate and parameter each, e.g ["AGE",     |
|                                                   | ("WGT", "CL")]. The associated effects are extracted from the given search space.                               |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``retries_strategy``                              | Decide how to use the retries tool. Valid options are 'skip', 'all_final' or 'final'. Default is 'all_final'    |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``seed``                                          | A random number generator or seed to use for steps with random sampling.                                        |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``parameter_uncertainty_method``                  | Parameter uncertainty method to use. Currently implemented methods are: 'SANDWICH', 'SMAT', 'RMAT' and 'EFIM'.  |
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

When running the tool with administration 'ivoral' with a dataset as input, the dataset is required to have a CMT column with values 1 
(oral doses) and 2 (IV doses). This is required for the creation of the initial one-compartment model with first order absorption. 
In order to easily differentiate the two doses, an administration ID (ADMID) column will be added to the data as well. This will be 
used in order to differentiate the different doses from one another with respect to the applied error model. If a model is used as 
input instead, this is not applied as it is assumed to have the correct CMT values for the connected model, along with a way of 
differentiating the doses from one another.

.. _search_space_amd:

~~~~~~~~~~~~
Search space
~~~~~~~~~~~~

A search space can be used to define all possible (and allowed) combinations of model features when searching for a
model. Currently, the search space support both structural as well as covariate models.
All features are given in the same MFL string. The different search spaces are then extracted from there and have no
effect on one another. 

If no search space is given for either the structural or covariate modeling, a default search space will be applied. This will
be based on the model type as well as administration. Please check the respective :ref:`model type page<modeltypes_amd>` to get
information on what is used for the specific model type/administration combination.

.. note::
    Please see the description of :ref:`mfl` for how to define the search space for the structural and covariate models.

Example
~~~~~~~

For a PK oral model, the default is:

.. code-block::

    ABSORPTION([FO,ZO,SEQ-ZO-FO])
    ELIMINATION(FO)
    LAGTIME([OFF,ON])
    TRANSITS([0,1,3,10],*)
    PERIPHERALS([0,1])
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

Only a single strategy can be used for each AMD run. Combinations of strategies are not
supported. However each of the subtools used in AMD is available to use manually as well.

.. note::
    Please note that the following is a general description of the different components executed
    by the AMD tool. Please see corresponding :ref:`model type page<modeltypes_amd>`
    for a detailed outline on how the different components are run.

default (default)
~~~~~~~~~~~~~~~~~

If no argument is specified, 'default' will be used as the default strategy. This will use
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


reevaluation
~~~~~~~~~~~~

The reevaluation strategy is an extension of the 'default' strategy. It is defined by the re-running
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
the Structural, IIVsearch and RUVsearch steps of the AMD tool. The workflow hence
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
the Structural, RUVsearch and IIVsearch steps of the AMD tool. The workflow hence
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
the RUVsearch, Structural and IIVsearch steps of the AMD tool. The workflow hence
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
        
~~~~~~~
Retries
~~~~~~~

The ``retries_strategy`` argument determines whether the retries tool will be used or not, and 
on which models. The different options and their description can be seen below. See :ref:`retries`
for more details about the tool.

+----------------------+------------------------------------------------------------------+
| Strategy             | Description                                                      |
+======================+==================================================================+
| ``'all_final'``      | Retries tool run on final model from each tool.                  |
+----------------------+------------------------------------------------------------------+
| ``'final'``          | Retries tool run only on final model from complete AMD workflow  |
+----------------------+------------------------------------------------------------------+
| ``'skip'``           | No retries are run for any models                                |
+----------------------+------------------------------------------------------------------+

When running the tool from AMD, the settings below will be used. If argument ``seed`` is set,
the chosen seed or random number generator will be used for the random sampling within the tool.

+---------------------------+----------------------------------------------------------------------------------------------------+
| Argument                  | Setting                                                                                            |
+===========================+====================================================================================================+
| ``number_of_candidates``  | 5                                                                                                  |
+---------------------------+----------------------------------------------------------------------------------------------------+
| ``fraction``              | 0.1                                                                                                |
+---------------------------+----------------------------------------------------------------------------------------------------+
| ``scale``                 | 'UCP'                                                                                              |
+---------------------------+----------------------------------------------------------------------------------------------------+
| ``use_initial_estimates`` | False                                                                                              |
+---------------------------+----------------------------------------------------------------------------------------------------+
| ``prefix_name``           | The name of the previously run tool                                                                |
+---------------------------+----------------------------------------------------------------------------------------------------+
| ``seed``                  | ``seed`` (As defined in :ref:`AMD options<amd_args>`)                                              |
+---------------------------+----------------------------------------------------------------------------------------------------+

~~~~~~~~~~~~~~~~~~~
Strategy components
~~~~~~~~~~~~~~~~~~~

The subtools that are run in each step, along with their respective arguments, are dependent on the model type given.
Below follows a general description of each of the steps. As different model types can perform the same step  
differently, please see the specific :ref:`model type page<modeltypes_amd>` for more details.

Structural
~~~~~~~~~~

This component of the AMD run is usually found in the beginning of a strategy and aims to find the best structural model for the
specified model type. Oftentimes including structural covariates along with the structure of the compartment system.

Structural covariates are user defined covariate effects that are not tested but rather forcefully added to the input model.
These effects are given within the search space in the following way:

.. code-block::

    COVARIATE(CL, WGT, POW)
    COVARIATE?(@IIV, @CATEGORICAL, *)

In this search space, the power covariate effect of WGT on CL is interpreted as a structural covariate (due to the missing "?")
while the other statement is interpreted as an exploratory covariate effect and will be explored in a later covsearch run.

The structural component of an AMD run is heavily dependent on which model type is being analyzed. It is possible that both :ref:`modelsearch`
and :ref:`structsearch` are used when developing the structural model, e.g. for TMDD and drug metabolite models, Modelsearch will develop
the PK structural model and Structsearch will develop the TMDD / drug metabolite model

IIVsearch
~~~~~~~~~

This subtool selects the IIV structure.  The tool will find both the number of IIVs and their covariance structure.
See :ref:`iivsearch` for more details about the tool.

Which IIVs that are being added is dependent on the model type. For example, for PKPD models, IIVs are only added to
the PD parameters within the model.

Residual
~~~~~~~~

This subtool selects the residual error model connected to the model. See :ref:`ruvsearch` for more details about the tool.


IOVsearch
~~~~~~~~~

This subtool selects the IOV structure and tries to remove corresponding IIVs if possible, see :ref:`iovsearch` for
more details about the tool. If no argument for ``occasion`` is given, this tool will not be run.

Allometry
~~~~~~~~~

This subtool applies allometry to clearance and volume parameters of the inputted model. 

.. note::
    Please note that if ``ignore_datainfo_fallback`` is set to ``True`` and no allometric variable is given, this tool
    will not be run. See :ref:`allometry` for more details about the tool. 

Covariates
~~~~~~~~~~

This subtool selects which covariate effects to apply, see :ref:`covsearch` for more details about the tool. 

Covariate effects for this stage are specified in the search space by specifying the effect with a "?",
as the following example suggests:

.. code-block::

    COVARIATE?(@IIV, @CATEGORICAL, *)

Covariate effects are split into two types at this stage. Mechanistic as well as exploratory covariate effects. Both
are to be tested for the model, but the mechanistic covariate effects will be tested in a separate initial covsearch run.
These covariates are specified using the ``mechanistic_covariates`` argument.

Given the mechanistic covariates ``mechanistic_covariates = [AGE, (CL,WGT)]``, the following search space would be
evaluated accordingly:

.. code-block::

    COVARIATE?([CL,V], [AGE, WGT], *)
    COVARIATE?(Q, WGT, *)
    
.. tabs::

    .. tab:: Mechanistic
   
        .. code-block::
        
            COVARIATE?([CL,V], AGE, *)
            COVARIATE?(CL, WGT, *)
   
    .. tab:: Exploratory
    
         .. code-block::

             COVARIATE?([V,Q], WGT, *)

.. note::
    Please note that if ``ignore_datainfo_fallback`` is set to ``True`` and no covariates are given, this tool will not be run.

The search space of effects given to this tool should include all possible (and allowed) covariate effects for the 
resulting model. This means that covariate effects that are a part of the input model but not the given search space
will be removed.

.. note::
    As allometric scaling can be interpreted as a power covariate effect, these effects will be added to the search space
    to avoid removing them during a covsearch run, if allometry was a part of the strategy.

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


.. pharmpy-execute::
   :hide-code:

   res.final_model_vpc_plot


~~~~~~~~
Examples
~~~~~~~~

TMDD
~~~~

Run AMD for a TMDD model:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools read_modelfit_results, run_structsearch

    start_model = read_model('path/to/model')

    res = run_amd(
                modeltype='tmdd',
                input=start_model,
                search_space='PERIPHERALS([1,2]);ELIMINATION([FO,ZO])',
                dv_types={'drug': 1, 'target': 2, 'complex': 3}
                )

.. note::
   The name of the DVID column must be "DVID".

PKPD
~~~~

Run AMD for a PKPD model:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools read_modelfit_results, run_structsearch

    start_model = read_model('path/to/model')

    res = run_amd(
                modeltype='pkpd',
                input=start_model,
                search_space='DIRECTEFFECT(*)',
                b_init=0.1,
                emax_init=1,
                ec50_init=0.1,
                met_init=0.4,
                )

.. note::
   The input model must be a PK model with a PKPD dataset. The name of the DVID column must be "DVID".
