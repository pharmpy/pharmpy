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
    res = run_iivsearch(algorithm='brute_force',
                        model=start_model,
                        results=start_model_results,
                        iiv_strategy='no_add',
                        rank_type='bic',
                        cutoff=None)

This will take an input model ``model`` and run the brute force ``algorithm``. IIVs on structural parameters
(such as mean absorption time) will not be added to the input model since ``iiv_strategy`` is set to be 'no_add'.
The candidate models will be ranked using ``bic`` with default ``cutoff``, which for BIC is none.

To run IIVsearch from the command line, the example code is redefined accordingly:

.. code::

    pharmpy run iivsearch path/to/model 'brute_force' --iiv_strategy 'no_add' --rank_type 'bic'

~~~~~~~~~
Arguments
~~~~~~~~~

+-----------------------------------------------+--------------------------------------------------------------------+
| Argument                                      | Description                                                        |
+===============================================+====================================================================+
| :ref:`algorithm<algorithms_iivsearch>`        | Algorithm to use (e.g. ``'brute_force'``)                          |
+-----------------------------------------------+--------------------------------------------------------------------+
| :ref:`iiv_strategy<iiv_strategies_iivsearch>` | If/how IIV should be added to start model (default is to not add)  |
+-----------------------------------------------+--------------------------------------------------------------------+
| :ref:`rank_type<ranking_iivsearch>`           | Which selection criteria to rank models on, e.g. OFV (default is   |
|                                               | BIC)                                                               |
+-----------------------------------------------+--------------------------------------------------------------------+
| :ref:`cutoff<ranking_iivsearch>`              | Cutoff for the ranking function, exclude models that are below     |
|                                               | cutoff (default is none)                                           |
+-----------------------------------------------+--------------------------------------------------------------------+
| ``model``                                     | Input model                                                        |
+-----------------------------------------------+--------------------------------------------------------------------+
| ``results``                                   | ModelfitResults of input model                                     |
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

+-----------------------------------+--------------------------------------------------------------------------------+
| Algorithm                         | Description                                                                    |
+===================================+================================================================================+
| ``'brute_force_no_of_etas'``      | Removes available IIV in all possible combinations                             |
+-----------------------------------+--------------------------------------------------------------------------------+
| ``'brute_force_block_structure'`` | Tests all combinations of covariance structures                                |
+-----------------------------------+--------------------------------------------------------------------------------+
| ``'brute_force'``                 | First runs ``'brute_force_no_of_etas'``, then                                  |
|                                   | ``'brute_force_block_structure'``                                              |
+-----------------------------------+--------------------------------------------------------------------------------+

Brute force search for number of IIVs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``brute_force_no_of_etas`` algorithm will create candidate models for all combinations of removed IIVs. It will
also create a naive pooled model meaning all the etas are fixed to 0. This can be useful in identifying local minima,
since all other candidate models should have a lower OFV than the naive pooled model (which doesn't have any
inter-individual variability).

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
        }

Brute force search for covariance structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``brute_force_block_structure`` algorithm will create candidates with all possible IIV variance and covariance
structures from the IIVs in the base model.

.. graphviz::

    digraph BST {
            node [fontname="Arial"];
            base [label="Base model"]
            s0 [label="[CL]+[V]+[MAT]"]
            s1 [label="[CL,V]+[MAT]"]
            s2 [label="[CL,MAT]+[V]"]
            s3 [label="[V,MAT]+[CL]"]
            s4 [label="[CL,V,MAT]"]

            base -> s0
            base -> s1
            base -> s2
            base -> s3
            base -> s4
        }

Full brute force search
~~~~~~~~~~~~~~~~~~~~~~~

The full ``brute_force`` search combines the brute force algorithm for choosing number of etas with the brute force
algorithm for the block structure, by first choosing the number of etas then the block structure.

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

~~~~~~~~~~~~~~~~~~~~~
The IIVsearch results
~~~~~~~~~~~~~~~~~~~~~

The results object contains various summary tables which can be accessed in the results object, as well as files in
.csv/.json format. The name of the selected best model (based on the input selection criteria) is also included.

Consider a iivsearch run:

.. pharmpy-code::

    res = run_iivsearch(algorithm='brute_force',
                        model=start_model,
                        results=start_model_results,
                        iiv_strategy='no_add',
                        rank_type='bic',
                        cutoff=None)


The ``summary_tool`` table contains information such as which feature each model candidate has, the difference to the
start model (in this case comparing BIC), and final ranking:

.. pharmpy-execute::
    :hide-code:

    from pharmpy.results import read_results
    res = read_results('tests/testdata/results/iivsearch_results.json')
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
