.. _modelrank:

=========
ModelRank
=========

The ModelRank tool is a general tool which filters models based on strictness criteria and ranks the remaining models.
If parameter uncertainty is provided in the strictness criteria, ModelRank will dynamically run models with the
specified parameter uncertainty method. This tool is used by all :ref:`amd` tools, but can also be used standalone.

~~~~~~~
Running
~~~~~~~

The ModelRank tool is available in Pharmpy/pharmr.

To initiate ModelRank in Python/R:

.. pharmpy-code::

    from pharmpy.tools import run_modelrank

    res = run_modelrank(models=models,        # Read in from tool or by scripting
                        results=results,      # Read in from tool or by scripting
                        ref_model=ref_model,  # One of read in models to be reference
                        strictness='minimization_successful or (rounding_errors and sigdigs>=0.1)',
                        rank_type='lrt',
                        alpha=0.05)

This will take an list of ``models`` and their corresponding ``results`` and use a ``ref_model`` to compare e.g.
OFV with. Only models that fulfills the ``strictness`` criteria will be ranked, and likelihood ratio test will be
performed as the ``rank_type``, using 0.05 as the p-value cutoff ``alpha``.

To activate the functionality where parameter uncertainty is dynamically run, include RSE in your strictness criteria,
for example:

.. pharmpy-code::

    from pharmpy.tools import run_modelrank

    res = run_modelrank(models=models,        # Read in from tool or by scripting
                        results=results,      # Read in from tool or by scripting
                        ref_model=ref_model,  # One of read in models to be reference
                        strictness='minimization_successful or (rounding_errors and sigdigs>=0.1) and rse < 0.5',
                        rank_type='lrt',
                        alpha=0.05)


Arguments
~~~~~~~~~
For a more detailed description of each argument, see their respective chapter on this page.

Mandatory
---------

+-------------------------------------------------+------------------------------------------------------------------+
| Argument                                        | Description                                                      |
+=================================================+==================================================================+
| ``models``                                      | Models to rank                                                   |
+-------------------------------------------------+------------------------------------------------------------------+
| ``results``                                     | Modelfit results of models to rank                               |
+-------------------------------------------------+------------------------------------------------------------------+
| ``ref_model``                                   | Reference model for e.g. LRT or calculating dOFV                 |
+-------------------------------------------------+------------------------------------------------------------------+

Optional
--------

+-------------------------------------------------+------------------------------------------------------------------+
| Argument                                        | Description                                                      |
+=================================================+==================================================================+
| ``strictness``                                  | Which :ref:`strictness` to filter models on (default is          |
|                                                 | ``'minimization_successful or (rounding_errors and sigdigs >=    |
|                                                 | 0.1)'``                                                          |
+-------------------------------------------------+------------------------------------------------------------------+
| ``rank_type``                                   | Which :ref:`selection_criteria` to rank models on (default is    |
|                                                 | OFV)                                                             |
+-------------------------------------------------+------------------------------------------------------------------+
| ``alpha``                                       | :math:`\alpha` for likelihood ration test                        |
+-------------------------------------------------+------------------------------------------------------------------+
| ``search_space``                                | :ref:`Search space<mfl>` for candidate models (only applicable   |
|                                                 | for mBIC as rank type)                                           |
+-------------------------------------------------+------------------------------------------------------------------+
| ``E``                                           | E-value (only applicable for mBIC as rank type)                  |
+-------------------------------------------------+------------------------------------------------------------------+
| ``parameter_uncertainty_method``                | Parameter uncertainty method to use if necessary for strictness  |
|                                                 | check                                                            |
+-------------------------------------------------+------------------------------------------------------------------+


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Running ModelRank without parameter uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the strictness criteria does not check parameter uncertainty, or if input models already have estimated uncertainty,
the rank tool will first filter the models based on strictness, then rank the remaining models. If likelihood ratio
test is used, it will be performed before ranking.

.. graphviz::

    digraph BST {
            node [fontname="Arial",shape="rect"];
            base [label="Models + results", shape="oval"]
            s0 [label="Filter on strictness"]
            s1 [label="Rank models"]
            s2 [label="Final model", shape="oval"]

            base -> s0
            s0 -> s1
            s1 -> s2

    }

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Running ModelRank with parameter uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the strictness criteria check parameter uncertainty, the rank tool will first filter the models based on the
strictness it can assess (e.g. minimization status) and rank the remaining models. It will then take the highest ranked
and rerun the model, estimating the parameter uncertainty. If the model passes the full strictness criteria, it will
be selected as the final model, otherwise it will take the next best model. It will continue either until a model
fulfills the full criteria, or until all models have failed the strictness criteria.



.. graphviz::

    digraph G {
      input [
        label = "Models and results";
        shape = oval;
      ];
      s0 [
        label = "Filter models based on strictness without RSE";
        shape = rect;
      ];
      s1 [
          label = "Rank models";
          shape = rect;
      ]
      s2 [
          label = "Top ranked model";
          shape = oval;
      ]

      s3 [
          label = "Run model with estimated RSE";
          shape = rect;
      ]
      s4 [
          label = "Model passes full strictness criteria?";
          shape = rect;
      ]

      s5 [
          label = "Next best model";
          shape = oval;
      ]
      final [
          label = "Final model and results";
          shape = oval;
      ]

      input -> s0;
      s0 -> s1;
      s1 -> s2;
      s2 -> s3;
      s3 -> s4;
      s4 -> s5 [label = "No"]
      s5 -> s3 [label = "Until no more models"]
      s4 -> final[label = "Yes"];

    }

~~~~~~~
Results
~~~~~~~

The results object contains various summary tables which can be accessed in the results object, as well as files in
.csv/.json format. The selected best model and its results is also included in the strictness critieria.

Consider a ModelSearch run:

.. pharmpy-code::

    res = run_modelsearch(
            model=start_model,
            results=start_res,
            search_space='ABSORPTION([FO,ZO]);PERIPHERALS([0,1]);LAGTIME([OFF,ON])',
            algorithm='exhaustive_stepwise',
            rank_type='bic')

This will run the ModelRank tool, if we read in that result object we can exlpore in more detail how the models were
ranked.

The ``summary_tool`` table contains information such as which feature each model candidate has, the difference to the
start model (in this case comparing BIC), and final ranking:

.. pharmpy-execute::
    :hide-code:

    from pharmpy.workflows.results import read_results
    res = read_results('tests/testdata/results/modelrank_results.json')
    res.summary_tool


If any models were run with parameter uncertainty, the tool will have a ``summary_models`` table, where you can find
information about the actual model runs, such as minimization status, estimation time, and parameter estimates. The
table is generated with :py:func:`pharmpy.tools.summarize_modelfit_results`.

The ``summary_strictness`` table contains information about whether strictness was fulfilled or not and more detail
about which part of the strictness criteria failed or not.

.. pharmpy-execute::
    :hide-code:

    res.summary_strictness

The ``summary_selection_criteria`` table contains information about the different components of the selection criteria,
such as penalty terms if using AIC/BIC, p-values and cutoff etc. if using LRT.

.. pharmpy-execute::
    :hide-code:

    res.summary_selection_criteria









