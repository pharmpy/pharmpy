============
Model Search
============

The Model search tool is a general tool to search for the best model given a base model and a search space of model features.

.. warning::

    This tool is currently under development. The API can and will change rapidly.

~~~~~~~
Running
~~~~~~~

The easiest way to start a model search run is to use the `run_modelsearch` function.

.. code:: python

    from pharmpy.tools.modelsearch import run_modelsearch

    run_modelsearch(base_model, algorithm, mfl, rankfunc='ofv')

This would start a search starting with `base_model` using the `algorithm` search algorithm, search space given by the `mfl` model feature language description and ranking candidate models using `ofv`.

~~~~~~~~~~~~~~~~
The search space
~~~~~~~~~~~~~~~~

The model feature search space is a set of possible combinations of model features that we would like to concider to be candidates for the search. This is input
to the tool in the form of a `Model Feature Language` string. The `MFL` is a domain specific language designed to describe model features and sets of model features in a concise way. See the detailed description below.

~~~~~~~~~~
Algorithms
~~~~~~~~~~

The tool can conduct the searching using different search algorithms. The current built in algorithms can be seen in the table below

+------------+-------------------------------------------------------------------+
| Algorithm  | Description                                                       |
+============+===================================================================+
| exhaustive | All possible combinations of the search space are tested          |
+------------+-------------------------------------------------------------------+
| stepwise   | Add one feature in each step.                                     |
|            | Select the best model to use as starting point for the next step. |
+------------+-------------------------------------------------------------------+

Exhaustive search
~~~~~~~~~~~~~~~~~

An exhaustive search will test all possible combinations of features in one big run.

.. code::

    ABSORPTION([FO, ZO])
    ELIMINATION([FO, ZO])

.. graphviz::

    digraph BST {
        node [fontname="Arial"];
        base [label="Base model"]
        s1 [label="ABSORPTION(FO)"]
        s2 [label="ABSORPTION(ZO)"]
        s3 [label="ELIMINATION(FO)"]
        s4 [label="ELIMINATION(ZO)"]
        s5 [label="ABSORPTION(FO);ELIMINATION(FO)"]
        s6 [label="ABSORPTION(FO);ELIMINATION(ZO)"]
        s7 [label="ABSORPTION(ZO);ELIMINATION(FO)"]
        s8 [label="ABSORPTION(ZO);ELIMINATION(ZO)"]
        base -> s1
        base -> s2
        base -> s3
        base -> s4
        base -> s5
        base -> s6
        base -> s7
        base -> s8
    }


Stepwise search
~~~~~~~~~~~~~~~

Here is an example of a search tree for the search space

.. code::

    ABSORPTION([FO, ZO])
    ELIMINATION([FO, ZO])

.. graphviz::

    digraph BST {
        node [fontname="Arial"];
        base [label="Base model"]
        s1_1 [style="bold", label="ABSORPTION(FO)"]
        s1_2 [label="ABSORPTION(ZO)"]
        s1_3 [label="ELIMINATION(FO)"]
        s1_4 [label="ELIMINATION(ZO)"]
        base -> s1_1
        base -> s1_2
        base -> s1_3
        base -> s1_4
        s2_2 [style="bold", label="ABSORPTION(FO);ELIMINATION(FO)"]
        s2_3 [label="ABSORPTION(FO);ELIMINATION(ZO)"]
        s1_1 -> s2_2
        s1_1 -> s2_3
    }

Nodes in bold was selected at each step. Initial estimates are updated between the steps.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comparing and ranking candidates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The supplied `rankfunc` will be used to compare a set of candidate models and rank them. The ranking will be used in different ways depending on the
selected algorithm. The following rank functions are available:

+------------+-------------------------------------------------------------------+
| Rankfunc   | Description                                                       |
+============+===================================================================+
| ofv        | ΔOFV. Do not rank candidates with dOFV < 3.84                     |
+------------+-------------------------------------------------------------------+
| aic        | ΔAIC. Do not rank candidates with dOFV < 3.84                     |
+------------+-------------------------------------------------------------------+
| bic        | ΔBIC. Do not rank candidates with dOFV < 3.84                     |
+------------+-------------------------------------------------------------------+


~~~~~~~~~~~~~~~~~~~~~~~~
The Model Search results
~~~~~~~~~~~~~~~~~~~~~~~~

The results contain a `runs` table with an overview of all model runs that were performed and which models were selected.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model feature language reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model feature language can be used to describe model features for one single model or an entire space of model features, i.e. descriptions for multiple models. The basic building block of MFL is the feature description. A feature description consists of the name of a feature category followed by a comma separated list of arguments within parentheses. For example:

.. code::

    ABSORPTION(FO)

Each feature description describe one or multiple features in the same category of features. Features of the same category are mutually exclusive and cannot be applied to the same model. Multiple model feature desciptions can be combined by separating them with either newline or semi-colon.

The following two examples are equivalent:

.. code::

    ABSORPTION(FO);ELIMINATION(ZO)

.. code::

    ABSORPTION(FO)
    ELIMINATION(ZO)

Option types
~~~~~~~~~~~~

MFL support the following types of options to feature descriptions

+---------------+------------------+-------------------------------------------------------+
| Type          | Example          | Description                                           |
+===============+==================+=======================================================+
| token or name | :code:`FO`       | The name of a feature within a category               |
+---------------+------------------+-------------------------------------------------------+
| number        | :code:`1`        | A non-negative integer                                |
+---------------+------------------+-------------------------------------------------------+
| range         | :code:`0..2`     | A range of numbers (endpoints are included)           |
+---------------+------------------+-------------------------------------------------------+
| wildcard      | :code:`*`        | All features of a category                            |
+---------------+------------------+-------------------------------------------------------+
| array         | :code:`[FO, ZO]` | Multiple tokens or numbers                            |
+---------------+------------------+-------------------------------------------------------+

Model features
~~~~~~~~~~~~~~

MFL support the following model features

+---------------+-------------------------------+-------------------------------------------------------+
| Category      | Options                       | Description                                           |
+===============+===============================+=======================================================+
| ABSORPTION    | :code:`FO, ZO, SEQ-ZO-FO`     | Absorption rate                                       |
+---------------+-------------------------------+-------------------------------------------------------+
| ELIMINATION   | :code:`FO, ZO, MM, MIX-FO-MM` | Elimination rate                                      |
+---------------+-------------------------------+-------------------------------------------------------+
| PERIPHERALS   | `number`                      | Number of peripheral compartments                     |
+---------------+-------------------------------+-------------------------------------------------------+
| TRANSITS      | `number`                      | Number of transit compartments                        |
+---------------+-------------------------------+-------------------------------------------------------+
| LAGTIME       | None                          | Lagtime                                               |
+---------------+-------------------------------+-------------------------------------------------------+

Redundant descriptions
~~~~~~~~~~~~~~~~~~~~~~

It is allowed to descripe the same feature multiple times. This will not make any difference for which features are described.

.. code::

    ABSORPTION(FO)
    ABSORPTION([FO, ZO])

is equivalent to

.. code::

    ABSORPTION([FO, ZO])

and

.. code::

    PERIPHERALS(1..2)
    PERIPHERALS(1)

is equivalent to

.. code::

    PERIPHERALS(1..2)

Examples
~~~~~~~~

Search through all available absorption rates

.. code::

    ABSORPTION(*)

Allow all combinations of absorption and elimination rates

.. code::

    ABSORPTION(*)
    ELIMINATION(*)

Consider 1, 2 and 3 peripheral compartments and none or upto 10 transit compartments:

.. code::

    PERIPHERALS(1..3)
    TRANSITS(0..10)
