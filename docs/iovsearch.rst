.. _iovsearch:

=========
IOVsearch
=========

The IOVsearch tool is a general tool to decide the best IOV structure given a start model. This includes adding IOV as
well as testing to remove associated IIVs.

~~~~~~~
Running
~~~~~~~

The IOVsearch tool is available both in Pharmpy/pharmr and from the command line.

To initiate IOVsearch in Python/R:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools import read_modelfit_results, run_iovsearch

    start_model = read_model('path/to/model')
    start_model_results = read_modelfit_results('path/to/model')
    res = run_iovsearch(model=start_model,
                        results=start_model_results,
                        column='OCC',
                        list_of_parameters=None,
                        distribution='same-as-iiv',
                        rank_type='bic',
                        cutoff=None)

This will take an input model ``model`` and use the ``column`` ``'OCC'`` as the occasion column. IOV will be tested on
parameters in ``list_of_parameters``, which when none will be all parameters with IIV. The IOVs will have the same
``distribution`` as the IIVs. The candidate models will be ranked using ``bic`` with default ``cutoff``, which for BIC
is none.

You can limit the search to only certain parameters by giving a list:

.. pharmpy-code::

    res = run_iovsearch(model=start_model,
                        results=start_model_results,
                        column='OCC',
                        list_of_parameters=['CL', 'V'])

To run IOVsearch from the command line, the example code is redefined accordingly:

.. code::

    pharmpy run iovsearch path/to/model --column 'OCC' --distribution 'same-as-iiv' --rank_type 'bic'

~~~~~~~~~
Arguments
~~~~~~~~~

+---------------------------------------------+----------------------------------------------------------------------+
| Argument                                    | Description                                                          |
+=============================================+======================================================================+
| ``column``                                  | Name of column in dataset to use as occasion column (default is      |
|                                             | `'OCC'`). Note that this only makes sense for discrete occasion data.|
+---------------------------------------------+----------------------------------------------------------------------+
| ``list_of_parameters``                      | List of parameters to test IOV on, if none all parameters with IIV   |
|                                             | will be tested (default)                                             |
+---------------------------------------------+----------------------------------------------------------------------+
| :ref:`distribution<distribution_iovsearch>` | Which distribution added IOVs should have (default is same as IIVs)  |
+---------------------------------------------+----------------------------------------------------------------------+
| :ref:`rank_type<ranking_iovsearch>`         | Which selection criteria to rank models on, e.g. OFV (default is     |
|                                             | BIC)                                                                 |
+---------------------------------------------+----------------------------------------------------------------------+
| :ref:`cutoff<ranking_iovsearch>`            | Cutoff for the ranking function, exclude models that are below       |
|                                             | cutoff (default is none)                                             |
+---------------------------------------------+----------------------------------------------------------------------+
| ``model``                                   | Start model                                                          |
+---------------------------------------------+----------------------------------------------------------------------+
| ``results``                                 | ModelfitResults of start model                                       |
+---------------------------------------------+----------------------------------------------------------------------+

~~~~~~~~~
Algorithm
~~~~~~~~~

The current algorithm uses a brute force approach where it first tests which IOVs should be used, then if any
associated IIVs can be removed.

.. graphviz::

    digraph G {
      draw [
        label = "Input model";
        shape = rect;
      ];
      add_iov [
        label = "Add IOV to all given parameters or all parameters with IIV";
        shape = rect;
      ];
      remove_iov [
          label = "Create candidates where each possible subset of IOV is removed";
          shape = rect;
      ]
      better_iov [
          label = "Any candidate better than input?";
          shape = rect;
      ]
      best_model_iov_no [
          label = "Select input model";
          shape = rect;
      ]

      best_model_iov_yes [
          label = "Select best candidate model";
          shape = rect;
      ]
      remove_iiv [
          label = "Create candidates where each possible subset\n of IIVs connected to IOV is removed";
          shape = rect;
      ]
      better_iiv [
          label = "Any candidate better than previous?";
          shape = rect;
      ]
      best_model_iiv_yes [
          label = "Select best candidate model";
          shape = rect;
      ]
      best_model_iiv_no [
          label = "Select model with all IIVs";
          shape = rect;
      ]
      done [
          label = "Best model";
          shape = rect;
      ]

      draw -> add_iov;
      add_iov -> remove_iov[label = "Fit model"];
      remove_iov -> better_iov[label = "Fit models"];

      better_iov -> best_model_iov_yes[label = "Yes"];
      better_iov -> best_model_iov_no [label = "No"];

      best_model_iov_no -> done;
      best_model_iov_yes -> remove_iiv[label = "Fit models"];

      remove_iiv -> better_iiv;
      better_iiv -> best_model_iiv_yes[label = "Yes"];
      better_iiv -> best_model_iiv_no[label = "No"];

      best_model_iiv_yes -> done;
      best_model_iiv_no -> done;
    }

.. _distribution_iovsearch:

~~~~~~~~~~~~~~~~~~~~~~~~
Distribution of new IOVs
~~~~~~~~~~~~~~~~~~~~~~~~

The ``distribution`` option determines how the added IOVs should be distributed. The different options are described
below.

+-------------------+-------------------------------------------------+
| Distribution      | Description                                     |
+===================+=================================================+
| ``'same-as-iiv'`` | Copies the distribution of IIV etas (default)   |
+-------------------+-------------------------------------------------+
| ``'disjoint'``    | Disjoint normal distributions                   |
+-------------------+-------------------------------------------------+
| ``'joint'``       | Joint normal distribution                       |
+-------------------+-------------------------------------------------+
| ``'explicit'``    | Explicit mix of joint and disjoint distribution |
+-------------------+-------------------------------------------------+

By default, or when specifying ``'same-as-iiv'``, you get the same covariance
structure for added IOVs as the one that already exists for IIVs. If you want a
different structure, you can specify ``'disjoint'`` to force zero covariance
between added IOVs, or ``'joint'`` to force nonzero covariance. To get full
control over the covariance you can specify ``'explicit'`` and give the
structure explicitly as in the following example:

.. pharmpy-code::

    res = run_iovsearch(model=start_model,
                        results=start_model_results,
                        column='OCC',
                        list_of_parameters=[['CL', 'V'], ['KA']],
                        distribution='explicit')

In this example, the newly added clearance (CL) and volume (V) IOVs will have
nonzero mutual covariance, but will have zero covariance with the absorption
constant (KA) IOV.


.. _ranking_iovsearch:

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
| ``'bic'``  | ΔBIC (random). Default is to rank all candidates if no cutoff is provided.        |
+------------+-----------------------------------------------------------------------------------+

Information about how BIC is calculated can be found in :py:func:`pharmpy.modeling.calculate_bic`.

~~~~~~~
Results
~~~~~~~

The results object contains various summary tables which can be accessed in the results object, as well as files in
.csv/.json format. The name of the selected best model (based on the input selection criteria) is also included.

Consider a IOVsearch run:

.. pharmpy-code::

    res = run_iovsearch(column='VISI',
                        model=start_model,
                        results=start_model_results,
                        list_of_parameters=None,
                        rank_type='bic',
                        cutoff=None,
                        distribution='same-as-iiv')


The ``summary_tool`` table contains information such as which feature each model candidate has, the difference to the
start model (in this case comparing BIC), and final ranking:

.. pharmpy-execute::
    :hide-code:

    from pharmpy.results import read_results
    res = read_results('tests/testdata/results/iovsearch_results.json')
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

