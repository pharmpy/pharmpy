.. _estmethod:

=========
Estmethod
=========

The Estmethod tool is a general tool to compare estimation methods and/or solvers for a given model.

~~~~~~~
Running
~~~~~~~

The Estmethod tool is available both in Pharmpy/pharmr and from the command line.

To initiate Estmethod in Python/R:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools import read_modelfit_results, run_estmethod

    start_model = read_model('path/to/model')
    start_model_results = read_modelfit_results('path/to/model')
    res = run_estmethod(algorithm='exhaustive',
                        model=start_model,
                        results=start_model_results,
                        methods='all',
                        solvers=['LSODA', 'LSODI'])

This will take an input model ``start_model``. The tool will use the 'exhaustive' ``algorithm`` and try all combinations
between all available ``methods`` and ``solvers`` LSODA and LSODI.

To run Estmethod from the command line, the example code is redefined accordingly:

.. code::

    pharmpy run estmethod path/to/model 'exhaustive' --methods 'all' --solvers 'LSODA LSODI'

Arguments
~~~~~~~~~
For a more detailed description of each argument, see their respective chapter on this page.

+-------------------------------------------------+------------------------------------------------------------------+
| Argument                                        | Description                                                      |
+=================================================+==================================================================+
| :ref:`algorithm<algorithms_estmethod>`          | Algorithm to use (e.g. ``'reduced'``                             |
+-------------------------------------------------+------------------------------------------------------------------+
| :ref:`methods<methods_estmethod>`               | Which methods to test (e.g. ``['IMP']``)                         |
+-------------------------------------------------+------------------------------------------------------------------+
| :ref:`solvers<solvers_estmethod>`               | Which solvers to test (e.g. ``['LSODA']``)                       |
+-------------------------------------------------+------------------------------------------------------------------+
| ``model``                                       | Start model                                                      |
+-------------------------------------------------+------------------------------------------------------------------+
| ``results``                                     | ModelfitResults of the start model                               |
+-------------------------------------------------+------------------------------------------------------------------+

.. _algorithms_estmethod:

~~~~~~~~~~
Algorithms
~~~~~~~~~~

There are a few ways Estmethod can test the different solvers/methods. The available algorithms can be seen in the table
below.

+------------------------------+-------------------------------------------------------------------------------------+
| Algorithm                    | Description                                                                         |
+==============================+=====================================================================================+
| ``'exhaustive'``             | All combinations are tested (cartesian product)                                     |
+------------------------------+-------------------------------------------------------------------------------------+
| ``'exhaustive_with_update'`` | All combinations are tested, but additionally creates candidates updated from FOCE  |
+------------------------------+-------------------------------------------------------------------------------------+
| ``'exhaustive_only_eval'``   | All combinations are tested, but only does evaluation                               |
+------------------------------+-------------------------------------------------------------------------------------+

Exhaustive
~~~~~~~~~~

The ``exhaustive`` algorithm works by creating all combinations of methods and solvers. The candidates will have one
estimation step and an evaluation step with IMP as method to be able to compare the candidates.

.. graphviz::

        digraph BST {
        node [fontname="Arial"];
        base [label="Base model"]
        s1 [label="IMP+LSODA"]
        s2 [label="IMP+LSODI"]
        s3 [label="LAPLACE+LSODA"]
        s4 [label="LAPLACE+LSODI"]
        base -> s1
        base -> s2
        base -> s3
        base -> s4
    }

The following table contains the setting for the estimation step:

+---------------------------+----------------------------------------------------------------------------------------+
| Setting                   | Value                                                                                  |
+===========================+========================================================================================+
| ``interaction``           | ``True``                                                                               |
+---------------------------+----------------------------------------------------------------------------------------+
| ``maximum_evaluations``   | ``9999``                                                                               |
+---------------------------+----------------------------------------------------------------------------------------+
| ``auto``                  | ``True``                                                                               |
+---------------------------+----------------------------------------------------------------------------------------+
| ``keep_every_nth_iter``   | ``10``                                                                                 |
+---------------------------+----------------------------------------------------------------------------------------+

Settings for evaluation step is the same as for estimation step, with the following additions:

+---------------------------+----------------------------------------------------------------------------------------+
| Setting                   | Value                                                                                  |
+===========================+========================================================================================+
| ``method``                | ``IMP``                                                                                |
+---------------------------+----------------------------------------------------------------------------------------+
| ``isample``               | ``100000``                                                                             |
+---------------------------+----------------------------------------------------------------------------------------+
| ``niter``                 | ``10``                                                                                 |
+---------------------------+----------------------------------------------------------------------------------------+


Exhaustive (with update)
~~~~~~~~~~~~~~~~~~~~~~~~

The ``exhaustive_with_update`` algorithm is similar to the ``exhaustive`` algorithm, but in addition to the candidate
models that the ``exhaustive`` algorithm create, it will also create a set of candidates that will use the final
estimates of a candidate with ``FOCE`` as the initial estimates.

.. graphviz::

    digraph BST {
        node [fontname="Arial"];
        base [label="Base model"]
        foce [label="FOCE"]
        s1 [label="IMP+LSODA"]
        s2 [label="IMP+LSODI"]
        s3 [label="LAPLACE+LSODA"]
        s4 [label="LAPLACE+LSODI"]
        base -> foce
        base -> s1
        base -> s2
        base -> s3
        base -> s4
        update [label="Update initial estimates"]
        foce -> update
        s5 [label="IMP+LSODA"]
        s6 [label="IMP+LSODI"]
        s7 [label="LAPLACE+LSODA"]
        s8 [label="LAPLACE+LSODI"]
        update -> s5
        update -> s6
        update -> s7
        update -> s8
    }

Settings are the same as for ``exhaustive``.

Exhaustive (only evaluation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``exhaustive_only_eval`` algorithm has the same algorithm as ``exhaustive``, but instead of estimating the
candidate models it only evaluates.

Settings are the same as for ``exhaustive`` evaluation step, where the method is the method being examined.

.. _methods_estmethod:

~~~~~~~
Methods
~~~~~~~

For a list of supported methods, see :py:func:`pharmpy.model.EstimationStep.supported_methods` (to test ``FOCE`` with
``LAPLACE``, simply specify ``LAPLACE`` as input argument in the tool).

.. _solvers_estmethod:

~~~~~~~
Solvers
~~~~~~~

For a list of supported solvers, see :py:func:`pharmpy.model.EstimationStep.supported_solvers`.

~~~~~~~~~~~~~~~~~~~~~
The Estmethod results
~~~~~~~~~~~~~~~~~~~~~

The results object contains various summary tables which can be accessed in the results object, as well as files in
.csv/.json format.

Consider a Estmethod run with the ``exhaustive`` algorithm and testing ``FO`` and ``LSODA``:

.. pharmpy-code::

    res = run_estmethod(algorithm='exhaustive',
                        model=start_model,
                        results=start_model_results,
                        methods=['FO', 'IMP'])

The ``summary_tool`` table contains information such as which feature each model candidate has, the OFV, estimation
runtime, and parent model:

.. pharmpy-execute::
    :hide-code:

    from pharmpy.results import read_results
    res = read_results('tests/testdata/results/estmethod_results.json')
    res.summary_tool

To see information about the actual model runs, such as minimization status, estimation time, and parameter estimates,
you can look at the ``summary_models`` table. The table is generated with
:py:func:`pharmpy.tools.summarize_modelfit_results`.

.. pharmpy-execute::
    :hide-code:

    res.summary_models

You can also see a summary of the settings that were used:

.. pharmpy-execute::
    :hide-code:

    res.summary_settings

Finally, you can see a summary of different errors and warnings in ``summary_errors``.
See :py:func:`pharmpy.tools.summarize_errors` for information on the content of this table.

.. pharmpy-execute::
    :hide-code:

    import pandas as pd
    pd.set_option('display.max_colwidth', None)
    res.summary_errors
