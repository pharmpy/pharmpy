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

    from pharmpy.modeling import read_model
    from pharmpy.tools import run_covsearch, read_modelfit_results

    start_model = read_model('path/to/model')
    start_model_results = read_modelfit_results('path/to/model')
    res = run_covsearch(model=start_model,
                        results=start_model_results,
                        search_space='COVARIATE?(@IIV, @CONTINUOUS, *); COVARIATE?(@IIV, @CATEGORICAL, CAT)',
                        algorithm='scm-forward-then-backward',
                        p_forward=0.05,
                        p_backward=0.01,
                        max_steps=5)

In this example, we attempt up to five forward steps of the Stepwise
Covariate Modeling (SCM) algorithm on the model ``start_model``. The p-value
threshold for these steps is 5% and the candidate search_space consists of all (*)
supported effects (multiplicative) of continuous covariates on parameters with IIV,
and a (multiplicative) categorical effect of categorical covariates on parameters
with IIV. Once we have identified the best model with this method, we attempt
up to ``k-1`` backward steps of the SCM algorithm on this model, where ``k`` is
the number of successful forward steps. The p-value threshold for the backward
steps is 1% and the effects that are candidate for removal are only the ones
that have been added by the forward steps.

To run COVsearch from the command line, the example code is redefined accordingly:

.. code::

    pharmpy run covsearch path/to/model --search_space 'COVARIATE?(@IIV, @CONTINUOUS, *); COVARIATE?(@IIV, @CATEGORICAL, CAT)' --algorithm scm-forward-then-backward --p_forward 0.05 --p_backward 0.01 --max_steps 5

~~~~~~~~~
Arguments
~~~~~~~~~

Mandatory
---------

+---------------------------------------------+-----------------------------------------------------------------------+
| Argument                                    | Description                                                           |
+=============================================+=======================================================================+
| ``model``                                   | Start model                                                           |
+---------------------------------------------+-----------------------------------------------------------------------+
| ``results``                                 | ModelfitResults of start model                                        |
+---------------------------------------------+-----------------------------------------------------------------------+
| ``search_space``                            | The candidate parameter-covariate                                     |
|                                             | :ref:`search_space<search_space_covsearch>` to search through         |
|                                             | (required)                                                            |
+---------------------------------------------+-----------------------------------------------------------------------+

Optional
--------

+---------------------------------------------+-----------------------------------------------------------------------+
| Argument                                    | Description                                                           |
+=============================================+=======================================================================+
| ``p_forward``                               | The p-value threshold for forward steps (default is `0.01`)           |
+---------------------------------------------+-----------------------------------------------------------------------+
| ``p_backward``                              | The p-value threshold for backward steps (default is `0.001`)         |
+---------------------------------------------+-----------------------------------------------------------------------+
| ``max_steps``                               | The maximum number of search algorithm steps to perform, or `-1`      |
|                                             | for no maximum (default).                                             |
+---------------------------------------------+-----------------------------------------------------------------------+
| ``algorithm``                               | The search :ref:`algorithm<algorithm_covsearch>` to use               |
|                                             | (default is `'scm-forward-then-backward'`)                            |
+---------------------------------------------+-----------------------------------------------------------------------+
| ``max_eval``                                | Limit the number of function evaluations to 3.1 times that of the     |
|                                             | base model. Default is False.                                         |
+---------------------------------------------+-----------------------------------------------------------------------+
| ``adaptive_scope_reduction``                | Stash all non-significant parameter-covariate effects at each step in |
|                                             | the forward search for later evaluation. As soon as all significant   |
|                                             | effects have been tested the stashed effects gets evaluated in a      |
|                                             | normal forward approach. Default is False                             |
+---------------------------------------------+-----------------------------------------------------------------------+
| ``strictness``                              | :ref:`Strictness<strictness>` criteria for model selection.           |
|                                             | Default is "minimization_successful or                                |
|                                             | (rounding_errors and sigdigs>= 0.1)"                                  |
+---------------------------------------------+-----------------------------------------------------------------------+

.. _search_space_covsearch:

~~~~~~~~~~~~
Search space
~~~~~~~~~~~~

There are two kinds of candidate effects that can be described through the model
feature language (:ref:`MFL<mfl>`). For instance, say that we want to have want to forcefully
add an exponential effect on volume through the weight covariate as well as testing
adding an exponential effect on clearance with age as covariate. In this case, the following
MFL specification can be used:

.. pharmpy-code::

    run_covsearch(
        ...
        search_space='COVARIATE(CL, WT, EXP);COVARIATE?(V,AGE,EXP)',
        ...
    )
    
.. note::
    :code:`COVARIATE(...)` represent structural covariates while :code:`COVARIATE?(...)` represent exploratory. 

The search space is specified by first writing the parameters, then the covariates of interest,
which effect, and, optionally, the operation to use for the covariate effect (`'*'`
(default) or `'+'`). If the operation is omitted, the default operation will be used.

The `MFL` also provides additional features such as automatically- or
manually-defined symbols. For instance the example above can be rewritten as

.. pharmpy-code::

    run_covsearch(
        ...
        effects='LET(CONTINUOUS, [AGE,WT]);COVARIATE?([CL, V], @CONTINUOUS, EXP)'
        ...
    )

Notice how multiple statements are separated by semicolons `;`.
Omitting declaration of continuous covariates allows to let Pharmpy
automatically derive which covariates should be referred to by `@CONTINUOUS`.
For instance,

.. pharmpy-code::

    run_covsearch(
        ...
        search_space='COVARIATE?([CL, V], @CONTINUOUS, EXP)'
        ...
    )

would test an exponential covariate effect on clearance and volume for each
continuous covariate.

.. note::
    Covariates that are already present in the model will be removed, unless they are also part of the search space. See :ref:`Algorithm<algorithm_covsearch>` for more.

More automatic symbols are available. They are described in the :ref:`MFL
symbols section<mfl_symbols>`.

Wildcards
---------

In addition to symbols, using a wildcard `\*` can help refer to computed list
of values. For instance the MFL sentence `COVARIATE?(*, *, *)` represents "All
continuous covariate effects of all covariates on all PK parameters".

+-------------+---------------------------------------------+
| Type        | Description of wildcard definition          |
+=============+=============================================+
| Covariate   | All covariates                              |
+-------------+---------------------------------------------+
| Effect      | All continuous effects                      |
+-------------+---------------------------------------------+
| Parameter   | All PK parameters                           |
+-------------+---------------------------------------------+

.. note::
    Wildcard for effects cannot be used with structural covariates as only a single
    effect can be added per covariate for a certain parameter.

.. _algorithm_covsearch:

~~~~~~~~~
Algorithm
~~~~~~~~~

The current default search algorithm `'scm-forward-then-backward'` consists of
the SCM method with forward steps followed by backward steps. The covariate effects that are added are 
dependent on the effects that are already present in the input model. All covariate effects that are 
initially part of the model but are *not* part of the search space will be removed before starting 
the search. Covariate effects that are part of both the search space *and* the model will be left in 
the model but are removed from the search space. In this initial stage, any structural covariates 
defined within the search space (see :ref:`search_space<search_space_covsearch>`) will be added as well. If any
filtration is done, a new "filtered_input_model" is created, otherwise the input model will be used.

.. note::
    If a filtered model is required, the changes made is reflected in its description. 

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
            s9 [label="Forward search best model"]
            s10 [label="RemoveEffect(V, WT, EXP)"]
            s11 [label="RemoveEffect(CL, SEX, CAT)"]
            s12 [label="Backward search best model"]

            base -> s0
            base -> s1
            base -> s2
            base -> s3
            s3 -> s4
            s3 -> s5
            s3 -> s6
            s4 -> s7
            s4 -> s8
            s4 -> s9
            s9 -> s10
            s9 -> s11
            s9 -> s12
        }

To skip the backward steps use search algorithm `'scm-forward'`.

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
            s9 [label="Forward search best model"]

            base -> s0
            base -> s1
            base -> s2
            base -> s3
            s3 -> s4
            s3 -> s5
            s3 -> s6
            s4 -> s7
            s4 -> s8
            s4 -> s9
        }

Adaptive scope reduction
------------------------

The adaptive scope reduction option integrates part of the SCM+ procedure within the covsearch tool. This option will
modify the forward search such that only significant effects will be transferred to the next step. Insignificant effects 
are stored away. The number of possible steps in this search is dependent on the ``max_steps`` argument. With the resulting
model from this search as input, a regular 'scm-forward' procedure is applied, using only the previously insignificant effects.
The number of possible steps in this procedure is also determined by the ``max_steps`` argument.

If 'scm-forward-then-backward' is used, a subsequent backward search will follow.

~~~~~~~
Results
~~~~~~~

The results object contains various summary tables which can be accessed in the results object, as well as files in
.csv/.json format. The name of the selected best model (based on the input selection criteria) is also included.

Consider a `covsearch` run:

.. pharmpy-code::

    res = run_covsearch(model=start_model, results=start_model_results,
                        search_space='COVARIATE?([CL, MAT, VC], [AGE, WT], EXP);COVARIATE?([CL, MAT, VC], [SEX], CAT)')


The ``summary_tool`` table contains information such as which feature each
model candidate has, the difference with the start model (in this case
comparing BIC), and final ranking:

.. pharmpy-execute::
    :hide-code:
    :hide-output:

    from pharmpy.workflows.results import read_results
    res = read_results('tests/testdata/results/covsearch_results.json')

.. pharmpy-execute::

    res.summary_tool

To see information about the actual model runs, such as minimization status,
estimation time, and parameter estimates, you can look at the
``summary_models`` table. The table is generated with
:py:func:`pharmpy.tools.summarize_modelfit_results`.

.. pharmpy-execute::

    res.summary_models

You can see a summary of different errors and warnings in ``summary_errors``.
See :py:func:`pharmpy.tools.summarize_errors` for information on the content
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
