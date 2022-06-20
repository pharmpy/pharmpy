.. _covsearch:

=========
COVsearch
=========

The COVsearch tool is a general tool to identify covariates that explain
some of the inter-individual variability.

~~~~~~~
Running
~~~~~~~

The COVsearch tool is available both in Pharmpy/pharmr and from the command line.

To initiate COVsearch in Python/R:

.. pharmpy-code::

    from pharmpy.tools import run_covsearch

    start_model = read_model('path/to/model')
    res = run_covsearch(algorithm='scm-forward',
                        model=start_model,
                        effects='COVARIATE(@IIV, @CONTINUOUS, *); COVARIATE(@IIV, @CATEGORICAL, CAT, *)',
                        p_forward=0.05,
                        max_steps=5)

In this example, we will attempt up to five forward steps of the Stepwise
Covariate Modeling (SCM) algorithm on the model ``start_model``. The p-value
threshold for theses steps if 5% and the candidate effects consists of all
supported effects (multiplicative) of continuous covariates on parameters with IIV,
and a multiplicative categorical effect of categorical covariates on parameters
with IIV.

To run COVsearch from the command line, the example code is redefined accordingly:

.. code::

    pharmpy run covsearch path/to/model --algorithm scm-forward --effects 'COVARIATE(@IIV, @CONTINUOUS, *); COVARIATE(@IIV, @CATEGORICAL, CAT, *)' --p_forward 0.05 --max_steps 5

~~~~~~~~~
Arguments
~~~~~~~~~

+---------------------------------------------+----------------------------------------------------------------------+
| Argument                                    | Description                                                          |
+=============================================+======================================================================+
| :ref:`effects<effects_covsearch>`           | The candidate effects to search through (required)                   |
+---------------------------------------------+----------------------------------------------------------------------+
| ``p_forward``                               | The p-value threshold for forward steps (default is `0.05`)          |
+---------------------------------------------+----------------------------------------------------------------------+
| ``max_steps``                               | The maximum number of search algorithm steps to perform, or `-1`     |
|                                             | for no maximum (default).                                            |
+---------------------------------------------+----------------------------------------------------------------------+
| :ref:`algorithm<algorithm_covsearch>`       | The search algorithm to use (default is `'scm-forward'`)             |
+---------------------------------------------+----------------------------------------------------------------------+
| ``model``                                   | Start model                                                          |
+---------------------------------------------+----------------------------------------------------------------------+

.. _effects_covsearch:

~~~~~~~
Effects
~~~~~~~

Candidate effects can be described in a variety of ways. The most basic way is
to give an explicit list of candidate effects to try:

.. pharmpy-code::

    run_covsearch(
        ...
        effects=[
            ['CL', 'SEX', 'CAT', '+'],
            ['CL', 'WT', 'EXP'],
            ['V', 'SEX', 'CAT', '+'],
            ['V', 'WT', 'EXP'],
        ],
        ...
    )

In each candidate, the first item is the name of the parameter (or its
corresponding IIV ETA), the second item is the name of the covariate, the
third item is the name of the effect (see
:py:func:`pharmpy.modeling.add_covariate_effect` for a list of available
effects), and the fourth item (optional) is the operator used to combine the
existing parameter expression with the effect (`'*'` (default) or `'+'`).

A more compact way to list candidate effects is through a list of cartesian
products. For instance the following list of candidate effects

.. pharmpy-code::

    run_covsearch(
        ...
        effects=[
            ['CL', 'AGE', 'EXP'],
            ['CL', 'WT', 'EXP'],
            ['V', 'AGE', 'EXP'],
            ['V', 'WT', 'EXP'],
        ],
        ...
    )

can be simplified to

.. pharmpy-code::

    run_covsearch(
        ...
        effects=[
            [['CL', 'V'], ['AGE', 'WT'], 'EXP'],
        ],
        ...
    )


Finally, the candidate effects can be defined through a domain-specifc language
(DSL) sentence. For instance, the example above can be given as

.. pharmpy-code::

    run_covsearch(
        ...
        effects='COVARIATE([CL, V], [AGE, WT], EXP)',
        ...
    )

This DSL also provides additional features such as automatic or manual
aliases. For instance the example above can be rewritten as

.. pharmpy-code::

    run_covsearch(
        ...
        effects='CONTINUOUS([AGE,WT]);COVARIATE([CL, V], @CONTINUOUS, EXP)'
        ...
    )

Notice how multiple statements are separated by semicolons `;`.
Omitting declaration of continuous covariates allows to let Pharmpy
automatically derive which covariates should be referred to by `@CONTINUOUS`.
For instance,

.. pharmpy-code::

    run_covsearch(
        ...
        effects='COVARIATE([CL, V], @CONTINUOUS, EXP)'
        ...
    )

would test an exponential covariate effect on clearance and volume for each
continuous covariate.

More aliases are available and described in the next section.

COVsearch DSL aliases
~~~~~~~~~~~~~~~~~~~~~

The DSL supports the following aliases:

+-----------------+-------------+----------------+---------------------------------------------+
| Alias           | Type        | Definition     | Description                                 |
+=================+=============+================+=============================================+
| `@IIV`          | Parameter   | auto           | All parameters with a corresponding IIV ETA |
+-----------------+-------------+----------------+---------------------------------------------+
| `\*`            | Parameter   | auto           | All PK parameters                           |
+-----------------+-------------+----------------+---------------------------------------------+
| `@ABSORPTION`   | Parameter   | auto or manual | All PK absorption parameters                |
+-----------------+-------------+----------------+---------------------------------------------+
| `@ELIMINATION`  | Parameter   | auto or manual | All PK elimination parameters               |
+-----------------+-------------+----------------+---------------------------------------------+
| `@DISTRIBUTION` | Parameter   | auto or manual | All PK distribution parameters              |
+-----------------+-------------+----------------+---------------------------------------------+
| `@CONTINUOUS`   | Covariate   | auto or manual | All continuous covariates                   |
+-----------------+-------------+----------------+---------------------------------------------+
| `@CATEGORICAL`  | Covariate   | auto or manual | All categorical covariates                  |
+-----------------+-------------+----------------+---------------------------------------------+
| `\*`            | Covariate   | auto           | All covariates                              |
+-----------------+-------------+----------------+---------------------------------------------+
| `\*`            | Effect      | auto           | All continuous effects                      |
+-----------------+-------------+----------------+---------------------------------------------+

Manual aliases can be defined via the syntax `ALIAS([...])`. For instance to
declare a list of absorption parameters use `ABSORPTION(KA)`.

For aliases that are both automatic and manual, the automatic definition of an
alias gets overriden as soon as a manual definition is used for the alias.


.. _algorithm_covsearch:

~~~~~~~~~
Algorithm
~~~~~~~~~

The current default (and only) search algorithm `'scm-forward'` consists in
forward steps of the Stepwise Covariate Modeling method.

.. graphviz::

    digraph BST {
            node [fontname="Arial"];
            base [label="Base model"]
            s0 [label="AddEffect(CL, SEX, CAT)"]
            s1 [label="AddEffect(CL, WT, EXP)"]
            s2 [label="AddEffect(V, SEX, CAT)"]
            s3 [label="AddEffect(V, WT, EXP)"]
            s4 [label="AddEffect(CL, SEX, CAT)"]
            s5 [label="AddEffect(CL, WT, EXP)"]
            s6 [label="AddEffect(V, SEX, CAT)"]
            s7 [label="AddEffect(CL, WT, EXP)"]
            s8 [label="AddEffect(V, SEX, CAT)"]

            base -> s0
            base -> s1
            base -> s2
            base -> s3
            s3 -> s4
            s3 -> s5
            s3 -> s6
            s4 -> s7
            s4 -> s8
        }


~~~~~~~
Results
~~~~~~~

The tool creates various summary tables which can be accessed in the results object,
as well as files in .csv/.json format.

Consider a `covsearch` run:

.. pharmpy-code::

    res = run_covsearch(model=start_model,
                        effects='COVARIATE([CL, MAT, VC], [AGE, WT], EXP);COVARIATE([CL, MAT, VC], [SEX], CAT)')


The ``summary_tool`` table contains information such as which feature each
model candidate has, the difference with the start model (in this case
comparing BIC), and final ranking:

.. pharmpy-execute::
    :hide-code:
    :hide-output:

    from pharmpy.results import read_results
    res = read_results('tests/testdata/results/covsearch_results.json')

.. pharmpy-execute::

    res.summary_tool

To see information about the actual model runs, such as minimization status,
estimation time, and parameter estimates, you can look at the
``summary_models`` table. The table is generated with
:py:func:`pharmpy.modeling.summarize_modelfit_results`.

.. pharmpy-execute::

    res.summary_models

A summary table of predicted influential individuals and outliers can be seen
in ``summary_individuals_count``. See
:py:func:`pharmpy.modeling.summarize_individuals_count_table` for information
on the content of this table.

.. pharmpy-execute::

    res.summary_individuals_count

You can see different individual statistics in ``summary_individuals``.  See
:py:func:`pharmpy.modeling.summarize_individuals` for information on the
content of this table.

.. pharmpy-execute::

    res.summary_individuals

You can see a summary of different errors and warnings in ``summary_errors``.
See :py:func:`pharmpy.modeling.summarize_errors` for information on the content
of this table.

.. pharmpy-execute::

    res.summary_errors


Finally, the results object provides the same attributes as
:ref:`provided by SCM <scm>`


.. pharmpy-execute::

    res.steps


.. pharmpy-execute::

    res.ofv_summary


.. pharmpy-execute::

    res.candidate_summary
