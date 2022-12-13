.. _amd:

=================================
Automatic Model Development (AMD)
=================================

The AMD tool is a general tool for fully automatic model development to decide the best model given either a dataset
or a starting model. The tool is a combination of the following tools: :ref:`modelsearch`, :ref:`iivsearch`,
:ref:`iovsearch`, :ref:`ruvsearch`, :ref:`allometry`, and :ref:`covsearch`.

~~~~~~~
Running
~~~~~~~

The AMD tool is available both in Pharmpy/pharmr.

To initiate AMD in Python/R:

.. pharmpy-code::

    from pharmpy.tools import run_amd

    dataset_path = 'path/to/dataset'
    order = ['structural', 'iivsearch', 'residual', 'iovsearch', 'allometry', 'covariates']
    res = run_amd(input=dataset_path,
                  modeltype='pk_oral',
                  order=order,
                  search_space='LET(CATEGORICAL, [SEX]); LET(CONTINUOUS, [AGE])',
                  allometric_variable='WGT',
                  occasion='VISI')

This will take a dataset as ``input``, where the ``modeltype`` has been specified to be a PK oral drug. AMD will search
for the best structural model, IIV structure, and residual model in the specified ``order``. We specify the column SEX
as a ``categorical`` covariate and AGE as a ``continuous`` covariate. Finally, we declare the WGT-column as our
``allometric_variable``, VISI as our ``occasion`` column.

~~~~~~~~~
Arguments
~~~~~~~~~

.. _amd_args:

+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| Argument                                          | Description                                                                                                     |
+===================================================+=================================================================================================================+
| :ref:`input<input_amd>`                           | Path to a dataset or start model object                                                                         |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``results``                                       | ModelfitResults if input is a model                                                                             |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``modeltype``                                     | Type of model to build. Either 'pk_oral' or 'pk_iv' (default is 'pk_oral')                                      |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``cl_init``                                       | Initial estimate for the population clearance (default is 0.01)                                                 |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``vc_init``                                       | Initial estimate for the central compartment population volume (default is 1)                                   |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``mat_init``                                      | Initial estimate for the mean absorption time (only for oral models, default is 0.1)                            |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| :ref:`search_space<search_space_amd>`             | MFL for search space of structural and covariate models (default depends on ``modeltype``)                      |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``lloq``                                          | Lower limit of quantification. LOQ data will be removed.                                                        |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| :ref:`order<order_amd>`                           | Run order of tools (default is ['structural', 'iivsearch', 'residual', 'iovsearch', 'allometry', 'covariates']) |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``allometric_variable``                           | Variable to use for allometry (default is name of column described as body weight)                              |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``occasion``                                      | Name of occasion column                                                                                         |
+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+

.. _input_amd:

~~~~~
Input
~~~~~

The AMD tool can use both a dataset and a model as input. If the input is a dataset (with corresponding
:ref:`datainfo file<datainfo>`), Pharmpy will create a model with the following attributes:

* Structural: one compartment, first order absorption (if ``model_type`` is ``'pk_oral'``), first order elimination
* IIV: CL and VC with covariance (``'pk_iv'``) or CL and VC with covariance and MAT (``'pk_oral'``)
* Residual: proportional error model
* Estimation steps: FOCE with interaction

If the input is a model, the model needs to be a PK model.

.. warning::
    The AMD tool, or more specifically the :ref:`modelsearch` tool, does not support NONMEM models with a CMT or RATE
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

    ABSORPTION([ZO,SEQ-ZO-FO])
    ELIMINATION([MM,MIX-FO-MM])
    LAGTIME()
    TRANSITS([1,3,10],*)
    PERIPHERALS(1)
    COVARIATE(@IIV, @CONTINUOUS, *)
    COVARIATE(@IIV, @CATEGORICAL, CAT)

For a PK IV model, the default is:

.. code-block::

    ELIMINATION([MM,MIX-FO-MM])
    PERIPHERALS([1,2])
    COVARIATE(@IIV, @CONTINUOUS, *)
    COVARIATE(@IIV, @CATEGORICAL, CAT)

Note that defaults are overriden selectively: structural model features
defaults will be ignored as soon as one structural model feature is explicitly
given, but the covariate model defaults will stay in place, and vice versa. For
instance, if one defines ``search_space`` as ``LAGTIME()``, the effective
search space will be as follows:

.. code-block::

    LAGTIME()
    COVARIATE(@IIV, @CONTINUOUS, *)
    COVARIATE(@IIV, @CATEGORICAL, CAT)

.. _order_amd:

~~~~~~~~~~~~~~~~~
Order of subtools
~~~~~~~~~~~~~~~~~

The order of the subtools is specified in the ``order`` argument. Consider the default order:

.. pharmpy-code::

    from pharmpy.tools import run_amd

    dataset_path = 'path/to/dataset'
    res = run_amd(input=dataset_path, order=None)

This yields the following workflow:

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

If you want to change the order, input a list of the desired order:

.. pharmpy-code::

    from pharmpy.tools import run_amd

    dataset_path = 'path/to/dataset'
    order = ['structural', 'residual', 'iivsearch', 'iovsearch', 'allometry', 'covariates']
    res = run_amd(input=dataset_path, order=order)

Here, the residual model will be decided before `iivsearch`, which will yield:

.. graphviz::

    digraph BST {
            node [fontname="Arial",shape="rect"];
            rankdir="LR";
            base [label="Input", shape="oval"]
            s0 [label="structural"]
            s1 [label="residual"]
            s2 [label="iivsearch"]
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

You can also run subsets of the subtools:

.. pharmpy-code::

    from pharmpy.tools import run_amd

    dataset_path = 'path/to/dataset'
    res = run_amd(input=dataset_path, order=['structural', 'iivsearch', 'residual'])


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

The default algorithms for six tools in amd can be seen in the table below. For more details regarding the settings
for each subtool, see the respective subheading.

+------------------+-------------------------------------------------------------------------------------------------+
| Tool             | Description                                                                                     |
+==================+=================================================================================================+
| modelsearch      | Search for best structural model for a PK model, includes absorption, distribution, and         |
|                  | elimination                                                                                     |
+------------------+-------------------------------------------------------------------------------------------------+
| iivsearch        | Search for best IIV structure, both in terms of number of IIVs to keep as well as covariance    |
|                  | structure                                                                                       |
+------------------+-------------------------------------------------------------------------------------------------+
| iovsearch        | Search for best IOV structure and remove IIVs explained by IOV                                  |
+------------------+-------------------------------------------------------------------------------------------------+
| ruvsearch        | Search for best residual error model, test IIV on RUV, power on RUV, combined error model, and  |
|                  | time-varying                                                                                    |
+------------------+-------------------------------------------------------------------------------------------------+
| allometry        | Test allometric scaling                                                                         |
+------------------+-------------------------------------------------------------------------------------------------+
| covsearch        | Test and identify covariate effects                                                             |
+------------------+-------------------------------------------------------------------------------------------------+

Structural
~~~~~~~~~~

This subtool selects the best structural model, see :ref:`modelsearch` for more details about the tool. The settings
that the AMD tool uses for this subtool can be seen in the table below.

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

IIVsearch
~~~~~~~~~

This subtool selects the IIV structure, see :ref:`iivsearch` for more details about the tool. The settings
that the AMD tool uses for this subtool can be seen in the table below.


+---------------+----------------------------------------------------------------------------------------------------+
| Argument      | Setting                                                                                            |
+===============+====================================================================================================+
| algorithm     | ``'brute_force'``                                                                                  |
+---------------+----------------------------------------------------------------------------------------------------+
| iiv_strategy  | ``'fullblock'``                                                                                    |
+---------------+----------------------------------------------------------------------------------------------------+
| rank_type     | ``'bic'`` (type: iiv)                                                                              |
+---------------+----------------------------------------------------------------------------------------------------+
| cutoff        | ``None``                                                                                           |
+---------------+----------------------------------------------------------------------------------------------------+

IOVsearch
~~~~~~~~~

This subtool selects the IOV structure and tries to remove corresponding IIVs if possible, see :ref:`iovsearch` for
more details about the tool. The settings that the AMD tool uses for this subtool can be seen in the table below.

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
that the AMD tool uses for this subtool can be seen in the table below.


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
that the AMD tool uses for this subtool can be seen in the table below.


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


Covariates
~~~~~~~~~~

This subtool selects which covariate effects to apply, see :ref:`covsearch` for more details about the tool. The
settings that the AMD tool uses for this subtool can be seen in the table below.

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

~~~~~~~
Results
~~~~~~~

The results object contains the final selected model and various summary tables, all of which can be accessed in the
results object as well as files in .csv/.json format.

The ``summary_tool`` table contains information such as which feature each model candidate has, the difference to the
start model (in this case comparing BIC), and final ranking:

.. pharmpy-execute::
    :hide-code:

    from pharmpy.results import read_results
    res = read_results('tests/testdata/results/amd_results.json')
    res.summary_tool

To see information about the actual model runs, such as minimization status, estimation time, and parameter estimates,
you can look at the ``summary_models`` table. The table is generated with
:py:func:`pharmpy.modeling.summarize_modelfit_results`.

.. pharmpy-execute::
    :hide-code:

    res.summary_models

A summary table of predicted influential individuals and outliers can be seen in ``summary_individuals_count``.
See :py:func:`pharmpy.tools.summarize_individuals_count_table` for information on the content of this table.

.. pharmpy-execute::
    :hide-code:

    res.summary_individuals_count

Finally, you can see a summary of any errors and warnings of the final selected model in ``summary_errors``.
See :py:func:`pharmpy.tools.summarize_errors` for information on the content of this table.

.. pharmpy-execute::
    :hide-code:

    import pandas as pd
    pd.set_option('display.max_colwidth', None)
    res.summary_errors


