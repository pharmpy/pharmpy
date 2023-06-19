.. _modelsearch:

===========
Modelsearch
===========

The Modelsearch tool is a general tool to decide the best structural model given a base model and a search space of
model features. The tool supports different algorithms and selection criteria.

~~~~~~~
Running
~~~~~~~

The modelsearch tool is available both in Pharmpy/pharmr and from the command line.

To initiate modelsearch in Python/R:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools import read_modelfit_results, run_modelsearch

    start_model = read_model('path/to/model')
    start_model_results = read_modelfit_results('path/to/model')
    res = run_modelsearch(search_space='PERIPHERALS(1);LAGTIME()',
                          algorithm='reduced_stepwise',
                          model=start_model,
                          results=start_model_results,
                          iiv_strategy='absorption_delay',
                          rank_type='bic',
                          cutoff=None)

This will take an input model ``model`` with ``search_space`` as the search space, meaning adding one peripheral
compartment and lagtime will be tried. The tool will use the 'reduced_stepwise' search ``algorithm``. IIVs on
structural parameters (such as mean absorption time) will not be added to candidates since ``iiv_strategy`` is
set to be 'absorption_delay'. The candidate models will have BIC as the ``rank_type`` with default ``cutoff``,
which for BIC is None/NULL.

To run modelsearch from the command line, the example code is redefined accordingly:

.. code::

    pharmpy run modelsearch path/to/model 'PERIPHERALS(1);LAGTIME()' 'reduced_stepwise' --iiv_strategy 'absorption_delay' --rank_type 'bic'

.. warning::

    Currently modelsearch does not support a CMT-column, make sure it is dropped before starting the tool.


Arguments
~~~~~~~~~
For a more detailed description of each argument, see their respective chapter on this page.

+-------------------------------------------------+------------------------------------------------------------------+
| Argument                                        | Description                                                      |
+=================================================+==================================================================+
| :ref:`search_space<the search space>`           | Search space to test                                             |
+-------------------------------------------------+------------------------------------------------------------------+
| :ref:`algorithm<algorithms_modelsearch>`        | Algorithm to use (e.g. ``'reduced_stepwise'``)                   |
+-------------------------------------------------+------------------------------------------------------------------+
| :ref:`rank_type<ranking_modelsearch>`           | Which selection criteria to rank models on, e.g. OFV (default is |
|                                                 | BIC)                                                             |
+-------------------------------------------------+------------------------------------------------------------------+
| :ref:`cutoff<ranking_modelsearch>`              | Cutoff for the ranking function, exclude models that are below   |
|                                                 | cutoff (default is None/NULL)                                    |
+-------------------------------------------------+------------------------------------------------------------------+
| :ref:`iiv_strategy<iiv_strategies_modelsearch>` | If/how IIV should be added to candidate models (default is to    |
|                                                 | add to absorption delay parameters)                              |
+-------------------------------------------------+------------------------------------------------------------------+
| ``model``                                       | Start model                                                      |
+-------------------------------------------------+------------------------------------------------------------------+
| ``results``                                     | ModelfitResults of the start model                               |
+-------------------------------------------------+------------------------------------------------------------------+

.. _the search space:

~~~~~~~~~~~~~~~~
The search space
~~~~~~~~~~~~~~~~

The model feature search space is a set of possible combinations of model features that will be applied and tested on
the input model. The supported features cover absorption, absorption delay, elimination, and distribution. The search
space is given as a string with a specific grammar, according to the `Model Feature Language` (MFL) (see :ref:`detailed description<mfl>`).

.. _algorithms_modelsearch:

~~~~~~~~~~
Algorithms
~~~~~~~~~~

The tool can conduct the model search using different algorithms. The available algorithms can be seen in the table
below.

+---------------------------+----------------------------------------------------------------------------------------+
| Algorithm                 | Description                                                                            |
+===========================+========================================================================================+
| ``'exhaustive'``          | All possible combinations of the search space are tested                               |
+---------------------------+----------------------------------------------------------------------------------------+
| ``'exhaustive_stepwise'`` | Add one feature in each step in all possible orders                                    |
+---------------------------+----------------------------------------------------------------------------------------+
| ``'reduced_stepwise'``    | Add one feature in each step in all possible orders. After each feature layer, choose  |
|                           | best model between models with same features                                           |
+---------------------------+----------------------------------------------------------------------------------------+

Common behaviours between algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feature combination exclusions
------------------------------

Some combinations of features are excluded in algorithms that are performed stepwise, the following combinations are
never run:

+-----------------------+-------------------+
| Feature A             | Feature B         |
+=======================+===================+
| ABSORPTION(ZO)        | TRANSITS          |
+-----------------------+-------------------+
| ABSORPTION(SEQ-ZO-FO) | TRANSITS          |
+-----------------------+-------------------+
| ABSORPTION(SEQ-ZO-FO) | LAGTIME           |
+-----------------------+-------------------+
| LAGTIME               | TRANSITS          |
+-----------------------+-------------------+

Additionally, peripheral compartments are always run sequentially, i.e. the algorithm will never add more than one
compartment at a given step. This is done in order to allow for better initial estimates from previous peripherals.

Exhaustive search
~~~~~~~~~~~~~~~~~

An ``exhaustive`` search will test all possible combinations of features in the search space. All candidate models will be
created simultaneously from the input model.

.. code::

    ABSORPTION(ZO)
    ELIMINATION(MM)
    PERIPHERALS(1)

.. graphviz::

    digraph BST {
        node [fontname="Arial"];
        base [label="Base model"]
        s1 [label="ABSORPTION(ZO)"]
        s2 [label="ELIMINATION(MM)"]
        s3 [label="PERIPHERALS(1)"]
        s4 [label="ABSORPTION(ZO);ELIMINATION(MM)"]
        s5 [label="ABSORPTION(ZO);PERIPHERALS(1)"]
        s6 [label="ELIMINATION(MM);PERIPHERALS(1)"]
        s7 [label="ABSORPTION(ZO);ELIMINATION(MM);PERIPHERALS(1)"]
        base -> s1
        base -> s2
        base -> s3
        base -> s4
        base -> s5
        base -> s6
        base -> s7
    }

Exhaustive stepwise search
~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``exhaustive_stepwise`` search applies features in a stepwise manner such that only one feature is changed at a time.
Between each step, the initial estimates from the new candidate model will be updated from the final estimates from the
previous step.

.. graphviz::

    digraph BST {
        node [fontname="Arial"];
        base [label="Base model"]
        s1 [label="ABSORPTION(ZO)"]
        s2 [label="ELIMINATION(MM)"]
        s3 [label="PERIPHERALS(1)"]
        s4 [label="ELIMINATION(MM)"]
        s5 [label="PERIPHERALS(1)"]
        s6 [label="ABSORPTION(ZO)"]
        s7 [label="PERIPHERALS(1)"]
        s8 [label="ABSORPTION(ZO)"]
        s9 [label="ELIMINATION(MM)"]
        s10 [label="PERIPHERALS(1)"]
        s11 [label="ELIMINATION(MM)"]
        s12 [label="PERIPHERALS(1)"]
        s13 [label="ABSORPTION(ZO)"]
        s14 [label="ELIMINATION(MM)"]
        s15 [label="ABSORPTION(ZO)"]
        base -> s1
        base -> s2
        base -> s3
        s1 -> s4
        s1 -> s5
        s2 -> s6
        s2 -> s7
        s3 -> s8
        s3 -> s9
        s4 -> s10
        s5 -> s11
        s6 -> s12
        s7 -> s13
        s8 -> s14
        s9 -> s15
    }

Reduced stepwise search
~~~~~~~~~~~~~~~~~~~~~~~
The ``reduced_stepwise`` search is similar to the exhaustive stepwise search, but after each layer it compares models with
the same features, where the compared models were obtained by adding the features in a different order. Next, the
algorithm uses the best model from each comparison as the basis for the next layer, where the subsequent feature is
added.

.. graphviz::

    digraph BST {
        node [fontname="Arial"];
        base [label="Base model"]
        s1 [label="ABSORPTION(ZO)"]
        s2 [label="ELIMINATION(MM)"]
        s3 [label="PERIPHERALS(1)"]
        s4 [label="ELIMINATION(MM)"]
        s5 [label="PERIPHERALS(1)"]
        s6 [label="ABSORPTION(ZO)"]
        s7 [label="PERIPHERALS(1)"]
        s8 [label="ABSORPTION(ZO)"]
        s9 [label="ELIMINATION(MM)"]
        s10 [label="Best model"]
        s11 [label="Best model"]
        s12 [label="Best model"]
        s13 [label="PERIPHERALS(1)"]
        s14 [label="ELIMINATION(MM)"]
        s15 [label="ABSORPTION(ZO)"]
        base -> s1
        base -> s2
        base -> s3
        s1 -> s4
        s1 -> s5
        s2 -> s6
        s2 -> s7
        s3 -> s8
        s3 -> s9
        s4 -> s10
        s6 -> s10
        s5 -> s11
        s8 -> s11
        s7 -> s12
        s9 -> s12
        s10 -> s13
        s11 -> s14
        s12 -> s15
    }


.. _iiv_strategies_modelsearch:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adding IIV to the candidate models during search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``iiv_strategy`` option determines whether or not IIV on the PK parameters should be added to the candidate models.
The different strategies can be seen here:

+------------------------+----------------------------------------------------------------------------------+
| Strategy               | Description                                                                      |
+========================+==================================================================================+
| ``'no_add'``           | No IIVs are added during the search                                              |
+------------------------+----------------------------------------------------------------------------------+
| ``'add_diagonal'``     | IIV is added to all structural parameters as diagonal                            |
+------------------------+----------------------------------------------------------------------------------+
| ``'fullblock'``        | IIV is added to all structural parameters, and all IIVs will be in a full block  |
+------------------------+----------------------------------------------------------------------------------+
| ``'absorption_delay'`` | IIV is added only to the absorption delay parameter (default)                    |
+------------------------+----------------------------------------------------------------------------------+

For more information regarding which parameters are counted as structural parameters, see
:py:func:`pharmpy.modeling.add_pk_iiv`.

.. _ranking_modelsearch:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comparing and ranking candidates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The supplied ``rank_type`` will be used to compare a set of candidate models and rank them. Each candidate model
will be compared to the input model. A cutoff may also be provided if the user does not want to use the default.
The following rank functions are available:

+------------+-----------------------------------------------------------------------------------+
| Rank type  | Description                                                                       |
+============+===================================================================================+
| ``'ofv'``  | ΔOFV. Default is to not rank candidates with ΔOFV < cutoff (default 3.84)         |
+------------+-----------------------------------------------------------------------------------+
| ``'aic'``  | ΔAIC. Default is to rank all candidates if no cutoff is provided.                 |
+------------+-----------------------------------------------------------------------------------+
| ``'bic'``  | ΔBIC (mixed). Default is to rank all candidates if no cutoff is provided.         |
+------------+-----------------------------------------------------------------------------------+

Information about how BIC is calculated can be found in :py:func:`pharmpy.modeling.calculate_bic`.

~~~~~~~~~~~~~~~~~~~~~~~
The Modelsearch results
~~~~~~~~~~~~~~~~~~~~~~~

The results object contains various summary tables which can be accessed in the results object, as well as files in
.csv/.json format. The name of the selected best model (based on the input selection criteria) is also included.

Consider a modelsearch run with the search space of zero order absorption and adding one peripheral compartment:

.. pharmpy-code::

    res = run_modelsearch(search_space='PERIPHERALS(1);LAGTIME()',
                          algorithm='reduced_stepwise',
                          model=start_model,
                          results=start_model_results,
                          iiv_strategy='absorption_delay',
                          rank_type='bic',
                          cutoff=None)

The ``summary_tool`` table contains information such as which feature each model candidate has, the difference to the
start model (in this case comparing BIC), and final ranking:

.. pharmpy-execute::
    :hide-code:

    from pharmpy.results import read_results
    res = read_results('tests/testdata/results/modelsearch_results.json')
    res.summary_tool

To see information about the actual model runs, such as minimization status, estimation time, and parameter estimates,
you can look at the ``summary_models`` table. The table is generated with
:py:func:`pharmpy.tools.summarize_modelfit_results`.

.. pharmpy-execute::
    :hide-code:

    res.summary_models

A summary table of predicted influential individuals and outliers can be seen in ``summary_individuals_count``.
See :py:func:`pharmpy.tools.summarize_individuals_count_table` for information on the content of this table.

.. pharmpy-execute::
    :hide-code:

    res.summary_individuals_count

You can see different individual statistics in ``summary_individuals``.
See :py:func:`pharmpy.tools.summarize_individuals` for information on the content of this table.

.. pharmpy-execute::
    :hide-code:

    res.summary_individuals

Finally, you can see a summary of different errors and warnings in ``summary_errors``.
See :py:func:`pharmpy.tools.summarize_errors` for information on the content of this table.

.. pharmpy-execute::
    :hide-code:

    import pandas as pd
    pd.set_option('display.max_colwidth', None)
    res.summary_errors
