===
IIV
===

The IIV tool is a general tool to decide the best IIV structure given a start model. This includes deciding which IIV
to keep and the covariance structure based on a chosen selection criteria.

~~~~~~~
Running
~~~~~~~

The iiv tool is available both in Pharmpy/pharmr and from the command line.

To initiate iiv in Python:

.. code:: python

    from pharmpy.modeling import run_tool

    start_model = read_model('path/to/model')
    run_tool('iiv',
             algorithm='brute_force_no_of_etas',
             model=start_model,
             iiv_strategy=0,
             rankfunc='bic',
             cutoff=None)

This will take an input model ``model`` and run the brute_force_no_of_etas ``algorithm``. The tool will add structural
IIVs to the start model according to according to ``iiv_strategy`` 0, where no IIVs are added. The candidate models
will be ranked using ``bic`` with default ``cutoff``, which for BIC is none.

To run iiv from the command line, the example code is redefined accordingly:

.. code::

    pharmpy run iiv path/to/model 'brute_force_no_of_etas' --iiv_strategy 0 --rankfunc 'bic'

~~~~~~~~~
Arguments
~~~~~~~~~

+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| Argument                                          | Description                                                                             |
+===================================================+=========================================================================================+
| :ref:`algorithm<Algorithms>`                      | Algorithm to use (e.g. brute_force_no_of_etas)                                          |
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

~~~~~~~~~~
Algorithms
~~~~~~~~~~

Different aspects of the IIV structure can be explored in the tool depending on which algorithm is chosen. The
available algorithms can be seen in the table below.

+-----------------------------------+-----------------------------------------------------+
| Algorithm                         | Description                                         |
+===================================+=====================================================+
| ``'brute_force_no_of_etas'``      | Removes available IIV in all possible combinations  |
+-----------------------------------+-----------------------------------------------------+
| ``'brute_force_block_structure'`` | Tests all combinations of covariance structures     |
+-----------------------------------+-----------------------------------------------------+

Brute force search for number of IIVs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This algorithm will create candidate models for all combinations of removed IIVs. It will also create a naive pooled
model.

.. graphviz::

    digraph BST {
            node [fontname="Arial"];
            base [label="Base model"]
            s0 [label="None"]
            s1 [label="ETA(1)"]
            s2 [label="ETA(2)"]
            s3 [label="ETA(3)"]
            s4 [label="ETA(1);ETA(2)"]
            s5 [label="ETA(1);ETA(3)"]
            s6 [label="ETA(2);ETA(3)"]
            s7 [label="ETA(1);ETA(2);ETA(3)"]

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
            s0 [label="Diagonal"]
            s1 [label="ETA(1)+ETA(2)"]
            s2 [label="ETA(1)+ETA(3)"]
            s3 [label="ETA(2)+ETA(3)"]
            s4 [label="ETA(1)+ETA(2)+ETA(3)"]
            s5 [label="Fullblock"]

            base -> s0
            base -> s1
            base -> s2
            base -> s3
            base -> s4
            base -> s5
        }

~~~~~~~~~~~~~~
IIV strategies
~~~~~~~~~~~~~~

The IIV strategy refers to if/how IIV should be added to the PK parameters of the input model. The different strategies
can be seen the corresponding chapter in :ref:`modelsearch<iiv_strategies>`.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comparing and ranking candidates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This system is the same as for modelsearch, see :ref:`here<ranking>`.
