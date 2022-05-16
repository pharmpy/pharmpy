.. _iivsearch:

=========
IIVsearch
=========

The IIV tool is a general tool to decide the best IIV structure given a start model. This includes deciding which IIV
to keep and the covariance structure based on a chosen selection criteria.

~~~~~~~
Running
~~~~~~~

The iiv tool is available both in Pharmpy/pharmr and from the command line.

To initiate iiv in Python:

.. pharmpy-code::

    from pharmpy.modeling import run_iivsearch

    start_model = read_model('path/to/model')
    res = run_iivsearch(algorithm='brute_force',
                        model=start_model,
                        iiv_strategy=0,
                        rankfunc='bic',
                        cutoff=None)

This will take an input model ``model`` and run the brute_force_no_of_etas ``algorithm``. The tool will add structural
IIVs to the start model according to according to ``iiv_strategy`` 0, where no IIVs are added. The candidate models
will be ranked using ``bic`` with default ``cutoff``, which for BIC is none.

To run iiv from the command line, the example code is redefined accordingly:

.. code::

    pharmpy run iivsearch path/to/model 'brute_force' --iiv_strategy 0 --rankfunc 'bic'

~~~~~~~~~
Arguments
~~~~~~~~~

+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| Argument                                          | Description                                                                             |
+===================================================+=========================================================================================+
| :ref:`algorithm<Algorithms>`                      | Algorithm to use (e.g. brute_force)                                                     |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| :ref:`iiv_strategy<IIV strategies>`               | If/how IIV should be added to start model (default is 0)                                |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| :ref:`rankfunc<Comparing and ranking candidates>` | Which selection criteria to rank models on, e.g. OFV (default is BIC)                   |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| :ref:`cutoff<Comparing and ranking candidates>`   | Cutoff for the ranking function, exclude models that are below cutoff (default is None) |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| :ref:`iiv_strategy<IIV strategies>`               | If/how IIV should be added to candidate models (default is 0)                           |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``model``                                         | Start model                                                                             |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+

.. _algorithms:

~~~~~~~~~~
Algorithms
~~~~~~~~~~

Different aspects of the IIV structure can be explored in the tool depending on which algorithm is chosen. The
available algorithms can be seen in the table below.

+-----------------------------------+---------------------------------------------------------------------------------+
| Algorithm                         | Description                                                                     |
+===================================+=================================================================================+
| ``'brute_force_no_of_etas'``      | Removes available IIV in all possible combinations                              |
+-----------------------------------+---------------------------------------------------------------------------------+
| ``'brute_force_block_structure'`` | Tests all combinations of covariance structures                                 |
+-----------------------------------+---------------------------------------------------------------------------------+
| ``'brute_force'``                 | First runs ``'brute_force_no_of_etas'``, then ``'brute_force_block_structure'`` |
+-----------------------------------+---------------------------------------------------------------------------------+

Brute force search for number of IIVs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This algorithm will create candidate models for all combinations of removed IIVs. It will also create a naive pooled
model.

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

This will try to create available IIV variance structures, including models with no covariance (only diagonal), and
covariance between all IIVs (full block).

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

The full brute force search combines the brute force algorithm for choosing number of etas with the brute force
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


.. _iiv strategies:

~~~~~~~~~~~~~~
IIV strategies
~~~~~~~~~~~~~~

The IIV strategy refers to if/how IIV should be added to the PK parameters of the input model. The different strategies
can be seen the corresponding chapter in :ref:`modelsearch<iiv_strategies>`.

.. _comparing and ranking candidates:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comparing and ranking candidates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This system is the same as for modelsearch, see :ref:`here<ranking>`.

~~~~~~~~~~~~~~~~~~~~~
The IIVsearch results
~~~~~~~~~~~~~~~~~~~~~

The results object contains the candidate models, the start model, and the selected best model (based on the input
selection criteria). The tool also creates various summary tables which can be accessed in the results object,
as well as files in .csv/.json format.

Consider a iivsearch run with the search space of zero order absorption and adding one peripheral compartment:

.. pharmpy-code::

    res = run_iivsearch(algorithm='brute_force',
                        model=start_model,
                        iiv_strategy=0,
                        rankfunc='bic',
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
:py:func:`pharmpy.modeling.summarize_modelfit_results`.

.. pharmpy-execute::
    :hide-code:

    import pandas as pd
    pd.set_option("display.max_columns", 10)
    res.summary_models

A summary table of predicted influential individuals and outliers can be seen in ``summary_individuals_count``.
See :py:func:`pharmpy.modeling.summarize_individuals_count_table` for information on the content of this table.

.. pharmpy-execute::
    :hide-code:

    res.summary_individuals_count

Finally, you can see different individual statistics ``summary_individuals``.
See :py:func:`pharmpy.modeling.summarize_individuals` for information on the content of this table.

.. pharmpy-execute::
    :hide-code:

    res.summary_individuals
