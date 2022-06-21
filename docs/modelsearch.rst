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

To initiate modelsearch in Python:

.. pharmpy-code::

    from pharmpy.tools import run_modelsearch

    start_model = read_model('path/to/model')
    res = run_modelsearch(search_space='ABSORPTION(ZO);PERIPHERALS(1)',
                          algorithm='exhaustive',
                          model=start_model,
                          iiv_strategy='no_add',
                          rank_type='bic',
                          cutoff=None)

This will take an input model ``model`` with ``search_space`` as the search space, meaning zero order absorption and adding one
peripheral compartment will be tried. The tool will use the ``exhaustive`` search algorithm. Structural IIVs will not be
added to candidates since ``iiv_strategy`` is set to be 'no_add'. The candidate models will be ranked using ``bic``
with default ``cutoff``, which for BIC is none.

To run modelsearch from the command line, the example code is redefined accordingly:

.. code::

    pharmpy run modelsearch path/to/model 'ABSORPTION(ZO);PERIPHERALS(1)' 'exhaustive' --iiv_strategy 'no_add' --rank_type 'bic'

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
|                                                 | cutoff (default is none)                                         |
+-------------------------------------------------+------------------------------------------------------------------+
| :ref:`iiv_strategy<iiv_strategies_modelsearch>` | If/how IIV should be added to candidate models (default is to    |
|                                                 | add to absorption delay parameters)                              |
+-------------------------------------------------+------------------------------------------------------------------+
| ``model``                                       | Start model                                                      |
+-------------------------------------------------+------------------------------------------------------------------+

.. _the search space:

~~~~~~~~~~~~~~~~
The search space
~~~~~~~~~~~~~~~~

The model feature search space is a set of possible combinations of model features that will be applied and tested on
the input model. The supported features cover absorption, absorption delay, elimination, and distribution. The search
space is given as a string with a specific grammar, according to the `Model Feature Language` (MFL) (see detailed
description :ref:`below<mfl>`).

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

Exhaustive search
~~~~~~~~~~~~~~~~~

An exhaustive search will test all possible combinations of features in one big run.

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
The exhaustive stepwise search applies features in a stepwise manner such that only one feature is changed at a time.

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

.. _feature combination exclusions:

Feature combination exclusions
------------------------------

Some combinations of features have been excluded in this algorithm, the following combinations are never run:

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

Reduced stepwise search
~~~~~~~~~~~~~~~~~~~~~~~
The reduced stepwise is similar to the exhaustive stepwise search, but after each layer it compares models with
the same features, where the compared models arrived at the features in a different order. Next, the algorithm sends
the best model from each comparison to the next layer, where the subsequent feature is added.

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

The same feature combinations as in the exhaustive stepwise algorithm will be excluded (described
:ref:`here<Feature combination exclusions>`)


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
| ``'absorption_delay'`` | IIV is added to the absorption delay parameter (default)                         |
+------------------------+----------------------------------------------------------------------------------+

.. _ranking_modelsearch:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comparing and ranking candidates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The supplied ``rank_type`` will be used to compare a set of candidate models and rank them. A cutoff may also be provided
if the user does not want to use the default. The following rank functions are available:

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

The results object contains the candidate models, the start model, and the selected best model (based on the input
selection criteria). The tool also creates various summary tables which can be accessed in the results object,
as well as files in .csv/.json format.

Consider a modelsearch run with the search space of zero order absorption and adding one peripheral compartment:

.. pharmpy-code::

    res = run_modelsearch('ABSORPTION(ZO);PERIPHERALS(1)',
                          'exhaustive',
                          model=start_model,
                          iiv_strategy='no_add',
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
:py:func:`pharmpy.modeling.summarize_modelfit_results`.

.. pharmpy-execute::
    :hide-code:

    res.summary_models

A summary table of predicted influential individuals and outliers can be seen in ``summary_individuals_count``.
See :py:func:`pharmpy.modeling.summarize_individuals_count_table` for information on the content of this table.

.. pharmpy-execute::
    :hide-code:

    res.summary_individuals_count

You can see different individual statistics in ``summary_individuals``.
See :py:func:`pharmpy.modeling.summarize_individuals` for information on the content of this table.

.. pharmpy-execute::
    :hide-code:

    res.summary_individuals

Finally, you can see a summary of different errors and warnings in ``summary_errors``.
See :py:func:`pharmpy.modeling.summarize_errors` for information on the content of this table.

.. pharmpy-execute::
    :hide-code:

    import pandas as pd
    pd.set_option('display.max_colwidth', None)
    res.summary_errors


.. _mfl:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model feature language (MFL) reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `MFL` is a domain specific language designed to describe model features and sets of model features in a concise way.
It can be used to describe model features for one single model or an entire space of model features, i.e. descriptions
for multiple models. The basic building block of MFL is the feature description. A feature description consists of the
name of a feature category followed by a comma separated list of arguments within parentheses. For example:

.. code::

    ABSORPTION(FO)

Each feature description describes one or multiple features of the same category, i.e. absorption, absorption delay,
elimination, and distribution. Features of the same category are mutually exclusive and cannot be applied to the same
model. Multiple model feature descriptions can be combined by separating them with either newline or a semi-colon.

The following two examples are equivalent:

.. code::

    ABSORPTION(FO);ELIMINATION(ZO)

.. code::

    ABSORPTION(FO)
    ELIMINATION(ZO)

Option types
~~~~~~~~~~~~

MFL support the following types of options to feature descriptions:

+---------------+------------------+-------------------------------------------------------+
| Type          | Example          | Description                                           |
+===============+==================+=======================================================+
| token or name | :code:`FO`       | The name of a feature within a category               |
+---------------+------------------+-------------------------------------------------------+
| number        | :code:`1`        | A non-negative integer                                |
+---------------+------------------+-------------------------------------------------------+
| range         | :code:`0..2`     | A range of numbers (endpoints are included)           |
+---------------+------------------+-------------------------------------------------------+
| wildcard      | :code:`*`        | All features of a category                            |
+---------------+------------------+-------------------------------------------------------+
| array         | :code:`[FO, ZO]` | Multiple tokens or numbers                            |
+---------------+------------------+-------------------------------------------------------+

Model features
~~~~~~~~~~~~~~

MFL support the following model features:

+---------------+-------------------------------+--------------------------------------------------------------------+
| Category      | Options                       | Description                                                        |
+===============+===============================+====================================================================+
| ABSORPTION    | :code:`FO, ZO, SEQ-ZO-FO`     | Absorption rate                                                    |
+---------------+-------------------------------+--------------------------------------------------------------------+
| ELIMINATION   | :code:`FO, ZO, MM, MIX-FO-MM` | Elimination rate                                                   |
+---------------+-------------------------------+--------------------------------------------------------------------+
| PERIPHERALS   | `number`                      | Number of peripheral compartments                                  |
+---------------+-------------------------------+--------------------------------------------------------------------+
| TRANSITS      | `number`, DEPOT/NODEPOT       | Number of absorption transit compartments. Whether convert depot   |
|               |                               | compartment into a transit compartment                             |
+---------------+-------------------------------+--------------------------------------------------------------------+
| LAGTIME       | None                          | Absorption lagtime                                                 |
+---------------+-------------------------------+--------------------------------------------------------------------+


Describe intervals
~~~~~~~~~~~~~~~~~~

It is possible to use ranges and arrays to describe the search space for e.g. transit and peripheral compartments.

To add 1, 2 and 3 peripheral compartments:

.. code::

    PERIPHERALS(1)
    PERIPHERALS(2)
    PERIPHERALS(3)

This is equivalent to:

.. code::

    PERIPHERALS(1..3)

As well as:

.. code::

    PERIPHERALS([1,2,3])

Redundant descriptions
~~~~~~~~~~~~~~~~~~~~~~

It is allowed to describe the same feature multiple times, however, this will not make any difference for which
features are described.

.. code::

    ABSORPTION(FO)
    ABSORPTION([FO, ZO])

This is equivalent to:

.. code::

    ABSORPTION([FO, ZO])

And:

.. code::

    PERIPHERALS(1..2)
    PERIPHERALS(1)

Is equivalent to:

.. code::

    PERIPHERALS(1..2)

Examples
~~~~~~~~

An example of a search space for PK models with oral data:

.. code::

    ABSORPTION([ZO,SEQ-ZO-FO])
    ELIMINATION([MM,MIX-FO-MM])
    LAGTIME()
    TRANSITS([1,3,10],*)
    PERIPHERALS(1)

An example of a search space for PK models with IV data:

.. code::

    ELIMINATION([MM,MIX-FO-MM])
    PERIPHERALS([1,2])


Search through all available absorption rates:

.. code::

    ABSORPTION(*)

Allow all combinations of absorption and elimination rates:

.. code::

    ABSORPTION(*)
    ELIMINATION(*)
