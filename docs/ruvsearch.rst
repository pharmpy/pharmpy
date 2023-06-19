.. _ruvsearch:

=========
ruvsearch
=========

The RUVsearch tool is a general tool to decide the best residual error model given a start model.

~~~~~~~
Running
~~~~~~~

The RUVsearch tool is available both in Pharmpy/pharmr and from the command line.

To initiate RUVsearch in Python/R:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools import read_modelfit_results, run_ruvsearch

    start_model = read_model('path/to/model')
    start_model_results = read_modelfit_results('path/to/model')
    res = run_ruvsearch(model=start_model, results=start_model_results)

To run RUVsearch from the command line, the example code is redefined accordingly:

.. code::

    pharmpy run ruvsearch path/to/model

~~~~~~~~~
Arguments
~~~~~~~~~

+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| Argument                                          | Description                                                                             |
+===================================================+=========================================================================================+
| ``groups``                                        | Number of groups to use for the time varying model (default is 4)                       |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``p_value``                                       | p-value for model selection (default is 0.05)                                           |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``skip``                                          | List of residual error models to not consider                                           |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``model``                                         | Start model                                                                             |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``results``                                       | ModelfitResults for the start model                                                     |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+

~~~~~~
Models
~~~~~~

The various residual error models tested by `RUVsearch` can be found in the table below together with links to detailed documentation of the
functions used to create them.

+------------------+----------------------------------------------------------------------------------------+
| Model name       | Function                                                                               | 
+==================+========================================================================================+
| ``IIV_on_RUV``   | :py:func:`set_iiv_on_ruv<pharmpy.modeling.set_iiv_on_ruv>`                             |
+------------------+----------------------------------------------------------------------------------------+
| ``power``        | :py:func:`set_power_on_ruv<pharmpy.modeling.set_power_on_ruv>`                         |
+------------------+----------------------------------------------------------------------------------------+
| ``combined``     | :py:func:`set_combined_error_model<pharmpy.modeling.set_combined_error_model>`         |
+------------------+----------------------------------------------------------------------------------------+
| ``time_varying`` | :py:func:`set_time_varying_error_model<pharmpy.modeling.set_time_varying_error_model>` |
+------------------+----------------------------------------------------------------------------------------+


~~~~~~~~~
Procedure
~~~~~~~~~

RUVsearch is doing modeling on the conditional weighted residual [Ibrahim]_ of the fit of the input model to quickly assess which residual
model to select. The input model is then updated with the new residual error model and fit to see if the selected residual error
model was indeed better. This is done multiple times to see if additional features of the residual error model should be added.
(Due to the similarity between the power error model and the combined error model, the combination of these two error models is not considered in the procedure.
E.g., if the power error model is selected in the current iteration, the combined error model will be skipped in the next iteration automatically.)

.. graphviz::

    digraph G {
      draw [
        label = "Input model";
        shape = rect;
      ];
      ruvsearch [
        label = "Run residual error models";
        shape = rect;
      ];
      select [
        label = "Select best model";
        shape = rect;
      ];
      update [
          label = "Update and run input model";
          shape = rect;
      ]
      better [
          label = "Significantly better?";
          shape = diamond;
      ]
      done [
          label = "Done";
          shape = rect;
      ]

      draw -> ruvsearch -> select -> update -> better;
      better -> done [label = "No"];
      better -> ruvsearch [label = "Yes (max 3 times)"]
    }

Selection is done using the likelihood ratio test and a default p-value of 0.05.

~~~~~~~~~~~~~~~~~~~~~
The RUVsearch results
~~~~~~~~~~~~~~~~~~~~~

The results object contains the start model, the residual error models based on the conditional weighted residual(CWRES) of the start model,
the candidate models, and the selected best model (based on the input selection criteria). The tool also creates various summary tables which can be accessed in the results object,
as well as files in .csv/.json format.

Consider a standard ruvsearch run:

.. pharmpy-code::

    res = run_ruvsearch(model=start_model, results=start_model_results)

The ``cwres_models`` table contains information on the residual error models, such as the iteration times, the changes of OFV compared to the base CWRES model and some key parameter estimates.

.. pharmpy-execute::
    :hide-code:

    from pharmpy.results import read_results
    res = read_results('tests/testdata/results/ruvsearch_results.json')
    res.cwres_models

The ``summary_tool`` table contains information such as which feature each model candidate has, the difference to the
start model, and final ranking:

.. pharmpy-execute::
    :hide-code:

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

Finally, you can see different individual statistics ``summary_individuals``.
See :py:func:`pharmpy.tools.summarize_individuals` for information on the content of this table.

.. pharmpy-execute::
    :hide-code:

    res.summary_individuals


.. [Ibrahim] Moustafa M. A. Ibrahim, Rikard Nordgren, Maria C. Kjellsson, Mats O. Karlsson. Model-Based Residual Post-Processing for Residual Model Identification. The AAPS Journal 2018 https://doi.org/10.1208/s12248-018-0240-7
