===========
modelsearch
===========

The modelsearch tool is a general tool to decide the best structural model given a base model and a search space of model features. The tool
supports different algorithms and selection criteria.

.. warning::

    This tool and its documentation is currently under development. The API may change.

~~~~~~~
Running
~~~~~~~

The modelsearch tool is available both in Pharmpy/pharmr and from the command line.

.. code:: python

    from pharmpy.modeling import run_tool

    start_model = load_example_model('pheno')
    run_tool('modelsearch',
             'ABSORPTION(ZO);ELIMINATION(ZO)',
             'exhaustive',
             iiv_strategy=0,
             rankfunc='ofv',
             cutoff=None,
             model=start_model)

This would start a search starting with an input model `model` with `mfl` as the search space using the `algorithm` search algorithm, where
structural IIVs will be added to candidates according to strategy 0. The candidate models will be ranked using `ofv` with default `cutoff`.

To run modelsearch from the command line:

.. code::

    pharmpy run modelsearch path/to/model 'ABSORPTION(ZO);ELIMINATION(ZO)' 'exhaustive' --rankfunc 'aic'

Arguments
~~~~~~~~~
For a more detailed description of each argument, see their respective chapter on this page.

+--------------+-------------------------------------------------------------------+
| Argument     | Description                                                       |
+==============+===================================================================+
| mfl          | Search space to test                                              |
+--------------+-------------------------------------------------------------------+
| algorithm    | Algorithm to use (e.g. exhaustive)                                |
+--------------+-------------------------------------------------------------------+
| rankfunc     | Which selection criteria to rank models on (e.g. OFV, AIC)        |
+--------------+-------------------------------------------------------------------+
| cutoff       | Cutoff for the ranking function (exclude models that are below)   |
+--------------+-------------------------------------------------------------------+
| iiv_strategy | If/how IIV should be added to candidate models                    |
+--------------+-------------------------------------------------------------------+
| model        | Start model                                                       |
+--------------+-------------------------------------------------------------------+


~~~~~~~~~~~~~~~~
The search space
~~~~~~~~~~~~~~~~

The model feature search space is a set of possible combinations of model features that will be applied and tested on the input model. Supported features cover
absorption, elimination, distribution, and delay. The search space is given as a string with a specific grammar, which is called `Model Feature Language`
(MFL). The `MFL` is a domain specific language designed to describe model features and sets of model features in a concise way. See the detailed description
below.

~~~~~~~~~~
Algorithms
~~~~~~~~~~

The tool can conduct the searching using different search algorithms. The current built in algorithms can be seen in the table below.

+---------------------+-------------------------------------------------------------------+
| Algorithm           | Description                                                       |
+=====================+===================================================================+
| exhaustive          | All possible combinations of the search space are tested          |
+---------------------+-------------------------------------------------------------------+
| exhaustive_stepwise | Add one feature in each step in all possible orders               |
+---------------------+-------------------------------------------------------------------+
| reduced_stepwise    | Add one feature in each step in all possible orders.              |
|                     | After each feature layer, choose best model between models        |
|                     | with same features                                                |
+---------------------+-------------------------------------------------------------------+

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

Exhaustive stepwise search
~~~~~~~~~~~~~~~~~~~~~~~~~~
An exhaustive stepwise search will apply features in a stepwise manner such that only one feature is changed at a time.

Some combinations of features have been excluded in this algorithm, the following combinations are never run:

+-----------------------+-------------------+
| Feature A             | Feature B         |
+=======================+===================+
| ABSORPTION(ZO)        | TRANSITS          |
+-----------------------+-------------------+
| ABSORPTION(SEQ-ZO-FO) | TRANSITS          |
+-----------------------+-------------------+
| ABSORPTION(SEQ-ZO-FO) | LAGTIME           |
+-----------------------+-------------------+
| LAGTIME               | TRANSITS          |
+-----------------------+-------------------+

Reduced stepwise search
~~~~~~~~~~~~~~~~~~~~~~~
The reduced stepwise is similar to the exhaustive stepwise search, but it will after each layer compare models with
the same features (but from different order) and only send the best model for the next transformations.

~~~~~~~~~~~~~~
IIV strategies
~~~~~~~~~~~~~~

The `iiv_strategy` option determines whether or not IIV on the PK parameters should be added to the candidate models.
The different strategies can be seen here:

+-----------+----------------------------------------------------------+
| Strategy  | Description                                              |
+===========+==========================================================+
| 0         | No IIVs are added during the search                      |
+-----------+----------------------------------------------------------+
| 1         | IIV is added to all structural parameters as diagonal    |
+-----------+----------------------------------------------------------+
| 2         | IIV is added to all structural parameters as full block  |
+-----------+----------------------------------------------------------+
| 3         | IIV is added to MDT parameters.                          |
+-----------+----------------------------------------------------------+

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comparing and ranking candidates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The supplied `rankfunc` will be used to compare a set of candidate models and rank them. A cutoff may also be provided
if the user does not want to use the default. The following rank functions are available:

+------------+---------------------------------------------------------------------------+
| Rankfunc   | Description                                                               |
+============+===========================================================================+
| ofv        | ΔOFV. Default is to not rank candidates with ΔOFV < cutoff (default 3.84) |
+------------+---------------------------------------------------------------------------+
| aic        | ΔAIC. Default is to rank all candidates if no cutoff is provided.         |
+------------+---------------------------------------------------------------------------+
| bic        | ΔBIC (mixed). Default is to rank all candidates if no cutoff is provided. |
+------------+---------------------------------------------------------------------------+

~~~~~~~~~~~~~~~~~~~~~~~~
The Model Search results
~~~~~~~~~~~~~~~~~~~~~~~~

The results object contains the candidate models, the start model, and the selected best model (based on the input
selection criteria). The tool will also create various summary tables which can be accessed in the results object,
as well as files in .csv/.json format. In those you can find information about the ranking and relevant features
(`summary_tool`), the estimated models (`summary_models`), and individuals in each model (`summary_individuals`).

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
