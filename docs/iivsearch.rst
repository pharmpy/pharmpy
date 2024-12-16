.. _iivsearch:

=========
IIVsearch
=========

The IIVsearch tool is a general tool to decide the best IIV structure given a start model. This includes deciding which IIV
to keep and the covariance structure based on a chosen selection criteria.

~~~~~~~
Running
~~~~~~~

The IIVsearch tool is available both in Pharmpy/pharmr and from the command line.

To initiate IIVsearch in Python/R:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools import read_modelfit_results, run_iivsearch

    start_model = read_model('path/to/model')
    start_model_results = read_model_results('path/to/model')
    res = run_iivsearch(model=start_model,
                        results=start_model_results,
                        algorithm='top_down_exhaustive',
                        iiv_strategy='fullblock',
                        rank_type='bic')

This will take an input ``model`` and run the top down exhaustive ``algorithm`` for the number of etas.
No IIVs on structural parameters (such as mean absorption time) will be added since the ``iiv_strategy``
is ``'no_add'``. The candidate models will be ranked using ``bic``.

To run IIVsearch from the command line, the example code is redefined accordingly:

.. code::

    pharmpy run iivsearch path/to/model --algorithm 'top_down_exhaustive' --iiv_strategy 'fullblock' --rank_type 'bic'

~~~~~~~~~
Arguments
~~~~~~~~~

Mandatory
---------

+-----------------------------------------------+--------------------------------------------------------------------+
| Argument                                      | Description                                                        |
+===============================================+====================================================================+
| ``model``                                     | Input model                                                        |
+-----------------------------------------------+--------------------------------------------------------------------+
| ``results``                                   | ModelfitResults of input model                                     |
+-----------------------------------------------+--------------------------------------------------------------------+
| ``algorithm``                                 | :ref:`Algorithm<algorithms_iivsearch>` to use                      |
|                                               | (e.g. ``'top_down_exhaustive'``)                                   |
+-----------------------------------------------+--------------------------------------------------------------------+

Optional
--------

+-----------------------------------------------+--------------------------------------------------------------------+
| Argument                                      | Description                                                        |
+===============================================+====================================================================+
| ``iiv_strategy``                              | If/how IIV should be added to start model (default is to not add). |
|                                               | See :ref:`iiv_strategies_iivsearch`                                |
+-----------------------------------------------+--------------------------------------------------------------------+
| ``rank_type``                                 | Which :ref:`selection criteria<ranking_iivsearch>` to rank models  | 
|                                               | on, e.g. OFV (default is BIC)                                      |
+-----------------------------------------------+--------------------------------------------------------------------+
| ``linearize``                                 | Decide whether or not to :ref:`linearize<linearize_iivsearch>`     |
|                                               | model before starting the search.                                  |
|                                               | See :ref:`linearization tool<linearize>` for more. Default value   |
|                                               | is False.                                                          |
+-----------------------------------------------+--------------------------------------------------------------------+
| ``cutoff``                                    | :ref:`Cutoff<ranking_iivsearch>` for the ranking function, exclude |
|                                               | models that are below cutoff (default is none)                     |
+-----------------------------------------------+--------------------------------------------------------------------+
| ``keep``                                      | List of IIVs to keep, either by parameter name or ETA name.        |
|                                               | Default is ["CL"]                                                  |
+-----------------------------------------------+--------------------------------------------------------------------+
| ``strictness``                                | :ref:`strictness<strictness>` criteria for model selection.        |
|                                               | Default is "minimization_successful or                             |
|                                               | (rounding_errors and sigdigs>= 0.1)"                               |
+-----------------------------------------------+--------------------------------------------------------------------+
| ``correlation_algorithm``                     | Specify if an algorithm different from the argument ``algorithm``  |
|                                               | should be used when searching for the correlation structure for    |
|                                               | the added IIVs. If not specified, the same algorithm will be used  |
|                                               | when searching for which IIVs to add as well as for the            |
|                                               | correlation structure.                                             |
+-----------------------------------------------+--------------------------------------------------------------------+


.. note::

    In this documentation, "base model" will be used to describe the model which all candidates are based on. Note
    that if you have set ``iiv_strategy`` to anything other than 'no_add', `this model will be different to the
    input model`. The term "base model" can thus be either the input model or a copy with added IIVs.


.. _algorithms_iivsearch:

~~~~~~~~~~
Algorithms
~~~~~~~~~~

Different aspects of the IIV structure can be explored in the tool depending on which algorithm is chosen. The
available algorithms can be seen in the table below.

.. note::

    If only ``algorithm`` is specified, the same will be applied to ``correlation_algorithm`` if possible.
    If not, please see description below which would be used.
    We recommend setting both arguments if specific algorithms are wanted.

+-------------------------------------+--------------------------------------------------------------------------------+
| Algorithm                           | Description                                                                    |
+=====================================+================================================================================+
| ``'top_down_exhaustive'``           | Removes available IIV in all possible combinations. The covariance structure   |
|                                     | search will search all possible IIV variance and covariance structure          |
+-------------------------------------+--------------------------------------------------------------------------------+
| ``'bottom_up_stepwise'``            | Iteratively adds all available IIV, one at a time. After each addition, the    |
|                                     | best model is selected. The algorithm stops when no better model was found     |
|                                     | after adding a new ETA. The following covariance structure search uses         |
|                                     | the same approach as 'top_down_exhaustive'                                     |
+-------------------------------------+--------------------------------------------------------------------------------+
| ``'skip'``                          | Set this argument if you are certain to skip either the search for number of   |
|                                     | etas (:code:`algorithm='skip'`) or to skip the search for the best covariance  |
|                                     | structure (:code:`correlation_algorithm='skip'`). However, if algorithm is set |
|                                     | to skip, then ``correlation_algorithm`` need to be set to a valid value.       |
+-------------------------------------+--------------------------------------------------------------------------------+


Top down exhaustive search
--------------------------

The ``top_down_exhaustive`` search combines the top down exhaustive approach for choosing number of etas with the brute force
algorithm for the block structure, by first choosing the number of etas then the block structure.

The ``top_down_exhaustive`` algorithm for choosing number of etas will create candidate models for all combinations of removed IIVs. It will
also create a naive pooled model meaning all the etas are fixed to 0. This can be useful in identifying local minima,
since all other candidate models should have a lower OFV than the naive pooled model (which doesn't have any
inter-individual variability).

For the covariance structure search, the ``top_down_exhaustive`` algorithm will create candidates with all possible IIV variance and 
covariance structures from the IIVs in the base model.

.. graphviz::

    digraph BST {
            node [fontname="Arial"];
            base [label="Base model"]
            s0 [label="Naive pooled"]
            s1 [label="[CL]"]
            s2 [label="[V]"]
            s3 [label="[MAT]"]
            s4 [label="[CL,V]"]
            s5 [label="[CL,MAT]"]
            s6 [label="[V,MAT]"]
            s7 [label="[CL,V,MAT]"]

            base -> s0
            base -> s1
            base -> s2
            base -> s3
            base -> s4
            base -> s5
            base -> s6
            base -> s7

            s8 [label="[CL]+[V]+[MAT]"]
            s9 [label="[CL,V]+[MAT]"]
            s10 [label="[CL,MAT]+[V]"]
            s11 [label="[V,MAT]+[CL]"]
            s12 [label="[CL,V,MAT]"]

            s7 -> s8
            s7 -> s9
            s7 -> s10
            s7 -> s11
            s7 -> s12

        }
        
Bottom up stepwise search
-------------------------

The ``bottom_up_stepwise`` algorithm differ from the ``top_down_exhaustive`` as the models are created
in iterative steps, each adding a single ETA. The algorithm will create a model with all possible IIVs and in the first step
it will remove all but one. This ETA will be on clearance (CL) if possible. If not, the first parameter in alphabetical order
will have an ETA. This model is then run and its results are used to update the initial estimates of the model. In the next step,
a candidate model is created for each remaining parameter that could have an ETA put on it. All models are run, and the best model
is chosen for the next step, updating the initial values once more.

The candidate models are then compared using the specified rank type and if no better model can be found, the algorithm stops.

However, this algorithm is not supported to run for the covariance structure search and thusly it will use the algorithm ``top_down_exhaustive``
for this step.

.. graphviz::

    digraph BST {
            node [fontname="Arial"];
            s1 [label="[CL]"]
            s2 [label="[CL,V]"]
            s3 [label="[CL,MAT]"]
            s4 [label="[CL,V,MAT]"]

            s1 -> s2
            s1 -> s3
            s2 -> s4
            
            s5 [label="[CL]+[V]+[MAT]"]
            s6 [label="[CL,V]+[MAT]"]
            s7 [label="[CL,MAT]+[V]"]
            s8 [label="[V,MAT]+[CL]"]
            s9 [label="[CL,V,MAT]"]
            
            s4 -> s5
            s4 -> s6
            s4 -> s7
            s4 -> s8
            s4 -> s9
            
        }

.. _iiv_strategies_iivsearch:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adding IIV to the start model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``iiv_strategy`` option determines whether or not IIV on the PK parameters should be added to the input model.
The different strategies can be seen here:

+------------------------+----------------------------------------------------------------------------------+
| Strategy               | Description                                                                      |
+========================+==================================================================================+
| ``'no_add'``           | Input model is kept as base model                                                |
+------------------------+----------------------------------------------------------------------------------+
| ``'add_diagonal'``     | Diagonal IIV is added to all structural parameters                               |
+------------------------+----------------------------------------------------------------------------------+
| ``'fullblock'``        | IIV is added to all structural parameters, and all IIVs will be in a full block  |
+------------------------+----------------------------------------------------------------------------------+


.. _ranking_iivsearch:

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
| ``'bic'``  | ΔBIC (iiv). Default is to rank all candidates if no cutoff is provided.           |
+------------+-----------------------------------------------------------------------------------+

Information about how BIC is calculated can be found in :py:func:`pharmpy.modeling.calculate_bic`.

.. _linearize_iivsearch:

~~~~~~~~~~~~~~~~~~~~~~
Linearization approach
~~~~~~~~~~~~~~~~~~~~~~

IIVsearch can be run with linearization. In this approach, a base model with all possible IIVs will first be created and
run in order to get the derivatives. Next, IIVsearch calls the linearize tool to linearize and run the model. All
subsequent candidate models in IIVsearch will be linearized and estimated. Once the best model of these candidates
have been selected, a delinearized version of the best candidate is created and estimated.

.. graphviz::

    digraph G {
      draw [
        label = "Input model";
        shape = rect;
      ];
      derivative [
        label = "Add IIVs for derivatives";
        shape = rect;
      ];
      linearize [
          label = "Linearize model";
          shape = rect;
      ]
      cands [
          label = "Create linearized candidates";
          shape = rect;
      ]
      best_cand [
          label = "Select best linearized model and delinearize";
          shape = rect;
      ]
      better [
          label = "Better than input model?";
          shape = rect;
      ]
      select_lin [
          label = "Select input";
          shape = rect;
      ]
      select_input [
          label = "Select candidate";
          shape = rect;
      ]
      done [
          label = "Best model";
          shape = rect;
      ]

      draw -> derivative;
      derivative -> linearize[label = "Fit model"];
      linearize -> cands[label = "Fit model"];
      cands -> best_cand[label = "Fit models"];
      best_cand -> better[label = "Fit model"];

      better -> select_input[label = "Yes"];
      better -> select_lin [label = "No"];

      select_input -> done;
      select_lin -> done;

    }


~~~~~~~~~~~~~~~~~~~~~
The IIVsearch results
~~~~~~~~~~~~~~~~~~~~~

The results object contains various summary tables which can be accessed in the results object, as well as files in
.csv/.json format. The name of the selected best model (based on the input selection criteria) is also included.

Consider a iivsearch run:

.. pharmpy-code::

    res = run_iivsearch(model=start_model,
                        results=start_model_results,
                        algorithm='td_brute_force',
                        iiv_strategy='no_add',
                        rank_type='bic')


The ``summary_tool`` table contains information such as which feature each model candidate has, the difference to the
start model (in this case comparing BIC), and final ranking:

.. pharmpy-execute::
    :hide-code:

    from pharmpy.workflows.results import read_results
    res = read_results('tests/testdata/results/iivsearch_results.json')
    res.summary_tool

To see information about the actual model runs, such as minimization status, estimation time, and parameter estimates,
you can look at the ``summary_models`` table. The table is generated with
:py:func:`pharmpy.tools.summarize_modelfit_results`.

.. pharmpy-execute::
    :hide-code:

    res.summary_models

Finally, you can see a summary of different errors and warnings in ``summary_errors``.
See :py:func:`pharmpy.tools.summarize_errors` for information on the content of this table.

.. pharmpy-execute::
    :hide-code:

    import pandas as pd
    pd.set_option('display.max_colwidth', None)
    res.summary_errors
