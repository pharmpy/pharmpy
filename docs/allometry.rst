.. _allometry:

=========
allometry
=========

The allometry tool is a simple tool to add allometric scaling to a model and run the scaled model.

~~~~~~~
Running
~~~~~~~

The allometry tool is available both in Pharmpy/pharmr and from the command line.

To initiate allometry in Python/R:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools import read_modelfit_results, run_allometry

    start_model = read_model('path/to/model')
    start_model_results = read_modelfit_results('path/to/model')
    res = run_allometry(model=start_model, results=start_model_results)

To run allometry from the command line, the example code is redefined accordingly:

.. code::

    pharmpy run allometry path/to/model

~~~~~~~~~
Arguments
~~~~~~~~~

+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| Argument                                          | Description                                                                             |
+===================================================+=========================================================================================+
| ``model``                                         | Pharmpy model                                                                           |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``results``                                       | ModelfitResults of model                                                                |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``allometric_variable``                           | Name of the variable to use for allometric scaling (default is WT)                      |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``reference_value``                               | Reference value for the allometric variable (default is 70)                             |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``parameters``                                    | Parameters to apply scaling to (default is all CL, Q and V parameters)                  |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``initials``                                      | Initial estimates for the exponents. (default is to use 0.75 for CL and Qs and 1 for Vs)|
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``lower_bounds``                                  | Lower bounds for the exponents. (default is 0 for all parameters)                       |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``upper_bounds``                                  | Upper bounds for the exponents. (default is 2 for all parameters)                       |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``fixed``                                         | Should the exponents be fixed or not. (default True)                                    |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+

~~~~~~~~~
Procedure
~~~~~~~~~

The allometry procedure is simple.

.. graphviz::

    digraph G {
      draw [
        label = "Input model";
        shape = rect;
      ];
      allometry [
        label = "Add allometric scaling to model";
        shape = rect;
      ];
      run [
        label = "Run";
        shape = rect;
      ];

      draw -> allometry -> run;
    }

No model selection is done.

~~~~~~~~~~~~~~~~~~~~~
The allometry results
~~~~~~~~~~~~~~~~~~~~~

To see information about the actual model runs, such as minimization status, estimation time, and parameter estimates,
you can look at the ``summary_models`` table. The table is generated with
:py:func:`pharmpy.tools.summarize_modelfit_results`.

.. pharmpy-execute::
    :hide-code:

    from pharmpy.results import read_results
    res = read_results('tests/testdata/results/allometry_results.json')
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
