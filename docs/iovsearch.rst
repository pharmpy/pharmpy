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

    from pharmpy.modeling import run_iovsearch

    start_model = read_model('path/to/model')
    res = run_iovsearch(model=start_model,
                        column='OCC',
                        list_of_parameters=None,
                        distribution='same-as-iiv',
                        rankfunc='bic',
                        cutoff=None)

This will take an input model ``model`` and use the ``column`` ``OCC`` as the occasion column. IOV will be tested on
all parameters with IIV according to ``list_of_parameters`` with the same ``distribution`` as the IIVs. The candidate
models will be ranked using ``bic`` with default ``cutoff``, which for BIC is none.

To run IOVsearch from the command line, the example code is redefined accordingly:

.. code::

    pharmpy run iovsearch path/to/model --column 'OCC' --distribution 'same-as-iiv' --rankfunc 'bic'

~~~~~~~~~
Arguments
~~~~~~~~~

+---------------------------------------------+----------------------------------------------------------------------+
| Argument                                    | Description                                                          |
+=============================================+======================================================================+
| ``column``                                  | Name of column in dataset to use as occasion column (default is      |
|                                             | `'OCC'`)                                                             |
+---------------------------------------------+----------------------------------------------------------------------+
| ``list_of_parameters``                      | List of parameters to test IOV on, if none all parameters with IIV   |
|                                             | will be tested (default)                                             |
+---------------------------------------------+----------------------------------------------------------------------+
| :ref:`distribution<distribution_iovsearch>` | Which distribution added IOVs should have (default is same as IIVs)  |
+---------------------------------------------+----------------------------------------------------------------------+
| :ref:`rankfunc<ranking_iovsearch>`          | Which selection criteria to rank models on, e.g. OFV (default is     |
|                                             | BIC)                                                                 |
+---------------------------------------------+----------------------------------------------------------------------+
| :ref:`cutoff<ranking_iovsearch>`            | Cutoff for the ranking function, exclude models that are below       |
|                                             | cutoff (default is none)                                             |
+---------------------------------------------+----------------------------------------------------------------------+
| ``model``                                   | Start model                                                          |
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
          label = "Create candidates where each possible subset\n of IIVs connected to IIV is removed";
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
| ``'disjoint'``    | Disjoint normal distribution                    |
+-------------------+-------------------------------------------------+
| ``'joint'``       | Joint normal distribution                       |
+-------------------+-------------------------------------------------+
| ``'explicit'``    | Explicit mix of joint and disjoint distribution |
+-------------------+-------------------------------------------------+


.. _ranking_iovsearch:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comparing and ranking candidates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This system is the same as for modelsearch, see :ref:`here<ranking_modelsearch>`.

~~~~~~~
Results
~~~~~~~
