.. _modelsearch:

===========
Modelsearch
===========

The Modelsearch tool is a general tool to decide the best structural model given a base model and a search space of
model features. The tool supports different algorithms and selection criteria.

~~~~~~~
Running
~~~~~~~

The modelsearch tool is available both in Pharmpy/pharmr and from the command line.

To initiate modelsearch in Python/R:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools import read_modelfit_results, run_modelsearch

    start_model = read_model('path/to/model')
    start_model_results = read_modelfit_results('path/to/model')
    res = run_modelsearch(search_space='PERIPHERALS(1);LAGTIME(ON)',
                          algorithm='reduced_stepwise',
                          model=start_model,
                          results=start_model_results,
                          iiv_strategy='absorption_delay',
                          rank_type='bic',
                          cutoff=None)

This will take an input model ``model`` with ``search_space`` as the search space, meaning adding one peripheral
compartment and lagtime will be tried. The tool will use the 'reduced_stepwise' search ``algorithm``. IIVs on
structural parameters (such as mean absorption time) will not be added to candidates since ``iiv_strategy`` is
set to be 'absorption_delay'. The candidate models will have BIC as the ``rank_type`` with default ``cutoff``,
which for BIC is None/NULL.

To run modelsearch from the command line, the example code is redefined accordingly:

.. code::

    pharmpy run modelsearch path/to/model 'PERIPHERALS(1);LAGTIME(ON)' 'reduced_stepwise' --iiv_strategy 'absorption_delay' --rank_type 'bic'

Arguments
~~~~~~~~~
For a more detailed description of each argument, see their respective chapter on this page.

Mandatory
---------

+-------------------------------------------------+------------------------------------------------------------------+
| Argument                                        | Description                                                      |
+=================================================+==================================================================+
| ``search_space``                                | :ref:`Search space<the search space>` to test                    |
+-------------------------------------------------+------------------------------------------------------------------+
| ``algorithm``                                   | :ref:`Algorithm<algorithms_modelsearch>`                         |
|                                                 | to use (e.g. ``'reduced_stepwise'``)                             |
+-------------------------------------------------+------------------------------------------------------------------+
| ``model``                                       | Start model                                                      |
+-------------------------------------------------+------------------------------------------------------------------+
| ``results``                                     | ModelfitResults of the start model                               |
+-------------------------------------------------+------------------------------------------------------------------+

Optional
--------

+-------------------------------------------------+------------------------------------------------------------------+
| Argument                                        | Description                                                      |
+=================================================+==================================================================+
| ``rank_type``                                   | Which :ref:`selection criteria<ranking_modelsearch>` to rank     |
|                                                 | models on, e.g. OFV (default is BIC)                             |
+-------------------------------------------------+------------------------------------------------------------------+
| ``cutoff``                                      | :ref:`Cutoff<ranking_modelsearch>` for the ranking function,     | 
|                                                 | exclude models that are below cutoff (default is None/NULL)      |                           
+-------------------------------------------------+------------------------------------------------------------------+
| ``iiv_strategy``                                | If/how IIV should be added to candidate models (default is to    |
|                                                 | add to absorption delay parameters).                             |
|                                                 | See :ref:`iiv_strategies_modelsearch`                            |
+-------------------------------------------------+------------------------------------------------------------------+
| ``strictness``                                  | :ref:`Strictness<strictness>` criteria for model selection.      |
|                                                 | Default is "minimization_successful or                           |
|                                                 | (rounding_errors and sigdigs>= 0.1)"                             |
+-------------------------------------------------+------------------------------------------------------------------+

.. _the search space:

~~~~~~~~~~~~~~~~
The search space
~~~~~~~~~~~~~~~~

The model feature search space is a set of all possible combinations of model features that is allowed for the final model. The supported 
features cover absorption, absorption delay, elimination, and distribution. The search space is given as a string with a specific 
grammar, according to the `Model Feature Language` (MFL) (see :ref:`detailed description<mfl>`). If an attribute is not given, the default
value for that attribute will be used as seen below. If the input model is not part of the given search space, a base model will be created. This is 
done by performing the least amount of transformations to the input model in order to make the base model a part of the given search 
space.

+---------------+-------------------------------+
| Category      | DEFAULT                       |
+===============+===============================+
| ABSORPTION    | :code:`INST`                  |
+---------------+-------------------------------+
| ELIMINATION   | :code:`FO`                    |
|               |                               |
+---------------+-------------------------------+
| PERIPHERALS   | :code:`0`                     |
+---------------+-------------------------------+
| TRANSITS      | :code:`0`, :code:`DEPOT`      |
|               |                               |
+---------------+-------------------------------+
| LAGTIME       | :code:`OFF`                   |
+---------------+-------------------------------+

The logical flow for the creation of the base model can be seen below. In summary, given an input model and a search space, the first step is 
to examine if the input model is a part of the search space. If so, the model features for the input model is filtered from the search space. 
As these are already present in the input model, they are not needed in the search space. After filtration, all transformations that are left will 
be examined. However, if the input model is not part of the search space, the base model is created by which will be part of the search space. 
Following this, the model features from the base model is filtered from the search space which leaves the transformations left to be examined.


.. graphviz::

    digraph G {
    splines = false
      input_model [
        label = "Input model";
      ];
      
      search_space [
        label = "Search space";
      ];
      
      input_ss [
        label = "Input + search space";
      ];
      
      filter_model [
        label = "Filter model features from search space";
      ];
      
      create_base [
        label = "Create a base";
      ];

      input_model -> input_ss;
      search_space -> input_ss;
      input_ss -> create_base [label = "Input model &notin; search space"];
      create_base -> filter_model;
      input_ss -> filter_model [label = "Input model &isin; search space"];
    }
    
Some examples of this workflow :

+---------------------------+------------------+---------------------+--------------------------+
| Search space              | Input model      | Base model          | Transformations to apply |
+===========================+==================+=====================+==========================+
| ABSORPTION([FO,ZO])       | ABSORPTION(FO)   | ABSORPTION(FO)      | ABSORPTION(ZO)           |
| ELIMINATION(FO)           | ELIMINATION(ZO)  | ELIMINATION(FO)     | PERIPHERALS(2)           |
| PERIPHERALS([1,2])        | TRANSITS(0)      | TRANSITS(0)         |                          |
|                           | PERIPHERALS(0)   | PERIPHERALS(1)      |                          |
|                           | LAGTIME(ON)      | LAGTIME(OFF)        |                          |
+---------------------------+------------------+---------------------+--------------------------+
| ABSORPTION(FO)            | ABSORPTION(FO)   | ABSORPTION(FO)      | TRANSITS(2)              |
| ELIMINATION(FO)           | ELIMINATION(ZO)  | ELIMINATION(FO)     |                          |
| TRANSITS([1,2])           | TRANSITS(0)      | TRANSITS(1)         |                          |
|                           | PERIPHERALS(2)   | PERIPHERALS(0)      |                          |
|                           | LAGTIME(OFF)     | LAGTIME(OFF)        |                          |
+---------------------------+------------------+---------------------+--------------------------+
| ABSORPTION([FO,ZO])       | ABSORPTION(FO)   | Not needed since    | ABSORPTION(ZO)           |
| ELIMINATION([FO,ZO,MM])   | ELIMINATION(FO)  | input model is part | ELIMINATION([ZO,MM])     |
| PERIPHERALS([0,1,2])      | TRANSITS(0)      | of search space     | PERIPHERALS([1,2]        |
| LAGTIME([OFF,ON])         | PERIPHERALS(0)   |                     | LAGTIME(ON)              |
|                           | LAGTIME(OFF)     |                     |                          |
+---------------------------+------------------+---------------------+--------------------------+

.. _algorithms_modelsearch:

~~~~~~~~~~
Algorithms
~~~~~~~~~~

The tool can conduct the model search using different algorithms. The available algorithms can be seen in the table
below.

+---------------------------+----------------------------------------------------------------------------------------+
| Algorithm                 | Description                                                                            |
+===========================+========================================================================================+
| ``'exhaustive'``          | All possible combinations of the search space are tested                               |
+---------------------------+----------------------------------------------------------------------------------------+
| ``'exhaustive_stepwise'`` | Add one feature in each step in all possible orders                                    |
+---------------------------+----------------------------------------------------------------------------------------+
| ``'reduced_stepwise'``    | Add one feature in each step in all possible orders. After each feature layer, choose  |
|                           | best model between models with same features                                           |
+---------------------------+----------------------------------------------------------------------------------------+

Common behaviours between algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feature combination exclusions
------------------------------

Some combinations of features are excluded in algorithms that are performed stepwise, the following combinations are
never run:

+-----------------------+-------------------+
| Feature A             | Feature B         |
+=======================+===================+
| ABSORPTION(ZO)        | TRANSITS          |
+-----------------------+-------------------+
| ABSORPTION(SEQ-ZO-FO) | TRANSITS          |
+-----------------------+-------------------+
| ABSORPTION(SEQ-ZO-FO) | LAGTIME(ON)       |
+-----------------------+-------------------+
| ABSORPTION(INST)      | LAGTIME(ON)       |
+-----------------------+-------------------+
| ABSORPTION(INST)      | TRANSITS          |
+-----------------------+-------------------+
| LAGTIME(ON)           | TRANSITS          |
+-----------------------+-------------------+

Additionally, peripheral compartments are always run sequentially, i.e. the algorithm will never add more than one
compartment at a given step. This is done in order to allow for better initial estimates from previous peripherals.

Exhaustive search
~~~~~~~~~~~~~~~~~

An ``exhaustive`` search will test all possible combinations of features in the search space. All candidate models will be
created simultaneously from the input model.

.. code::

    ABSORPTION(ZO)
    ELIMINATION(MM)
    PERIPHERALS(1)

.. graphviz::

    digraph BST {
        node [fontname="Arial"];
        base [label="Base model"]
        s1 [label="ABSORPTION(ZO)"]
        s2 [label="ELIMINATION(MM)"]
        s3 [label="PERIPHERALS(1)"]
        s4 [label="ABSORPTION(ZO);ELIMINATION(MM)"]
        s5 [label="ABSORPTION(ZO);PERIPHERALS(1)"]
        s6 [label="ELIMINATION(MM);PERIPHERALS(1)"]
        s7 [label="ABSORPTION(ZO);ELIMINATION(MM);PERIPHERALS(1)"]
        base -> s1
        base -> s2
        base -> s3
        base -> s4
        base -> s5
        base -> s6
        base -> s7
    }

Exhaustive stepwise search
~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``exhaustive_stepwise`` search applies features in a stepwise manner such that only one feature is changed at a time.
Between each step, the initial estimates from the new candidate model will be updated from the final estimates from the
previous step.

.. graphviz::

    digraph BST {
        node [fontname="Arial"];
        base [label="Base model"]
        s1 [label="ABSORPTION(ZO)"]
        s2 [label="ELIMINATION(MM)"]
        s3 [label="PERIPHERALS(1)"]
        s4 [label="ELIMINATION(MM)"]
        s5 [label="PERIPHERALS(1)"]
        s6 [label="ABSORPTION(ZO)"]
        s7 [label="PERIPHERALS(1)"]
        s8 [label="ABSORPTION(ZO)"]
        s9 [label="ELIMINATION(MM)"]
        s10 [label="PERIPHERALS(1)"]
        s11 [label="ELIMINATION(MM)"]
        s12 [label="PERIPHERALS(1)"]
        s13 [label="ABSORPTION(ZO)"]
        s14 [label="ELIMINATION(MM)"]
        s15 [label="ABSORPTION(ZO)"]
        base -> s1
        base -> s2
        base -> s3
        s1 -> s4
        s1 -> s5
        s2 -> s6
        s2 -> s7
        s3 -> s8
        s3 -> s9
        s4 -> s10
        s5 -> s11
        s6 -> s12
        s7 -> s13
        s8 -> s14
        s9 -> s15
    }

Reduced stepwise search
~~~~~~~~~~~~~~~~~~~~~~~
The ``reduced_stepwise`` search is similar to the exhaustive stepwise search, but after each layer it compares models with
the same features, where the compared models were obtained by adding the features in a different order. Next, the
algorithm uses the best model from each comparison as the basis for the next layer, where the subsequent feature is
added.

.. graphviz::

    digraph BST {
        node [fontname="Arial"];
        base [label="Base model"]
        s1 [label="ABSORPTION(ZO)"]
        s2 [label="ELIMINATION(MM)"]
        s3 [label="PERIPHERALS(1)"]
        s4 [label="ELIMINATION(MM)"]
        s5 [label="PERIPHERALS(1)"]
        s6 [label="ABSORPTION(ZO)"]
        s7 [label="PERIPHERALS(1)"]
        s8 [label="ABSORPTION(ZO)"]
        s9 [label="ELIMINATION(MM)"]
        s10 [label="Best model"]
        s11 [label="Best model"]
        s12 [label="Best model"]
        s13 [label="PERIPHERALS(1)"]
        s14 [label="ELIMINATION(MM)"]
        s15 [label="ABSORPTION(ZO)"]
        base -> s1
        base -> s2
        base -> s3
        s1 -> s4
        s1 -> s5
        s2 -> s6
        s2 -> s7
        s3 -> s8
        s3 -> s9
        s4 -> s10
        s6 -> s10
        s5 -> s11
        s8 -> s11
        s7 -> s12
        s9 -> s12
        s10 -> s13
        s11 -> s14
        s12 -> s15
    }


.. _iiv_strategies_modelsearch:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adding IIV to the candidate models during search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``iiv_strategy`` option determines whether or not IIV on the PK parameters should be added to the candidate models.
The different strategies can be seen here:

+------------------------+----------------------------------------------------------------------------------+
| Strategy               | Description                                                                      |
+========================+==================================================================================+
| ``'no_add'``           | No IIVs are added during the search                                              |
+------------------------+----------------------------------------------------------------------------------+
| ``'add_diagonal'``     | IIV is added to all structural parameters as diagonal                            |
+------------------------+----------------------------------------------------------------------------------+
| ``'fullblock'``        | IIV is added to all structural parameters, and all IIVs will be in a full block  |
+------------------------+----------------------------------------------------------------------------------+
| ``'absorption_delay'`` | IIV is added only to the absorption delay parameter (default)                    |
+------------------------+----------------------------------------------------------------------------------+

For more information regarding which parameters are counted as structural parameters, see
:py:func:`pharmpy.modeling.add_pk_iiv`.

.. _ranking_modelsearch:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comparing and ranking candidates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The supplied ``rank_type`` will be used to compare a set of candidate models and rank them. Each candidate model
will be compared to the input model. A cutoff may also be provided if the user does not want to use the default.
The following rank functions are available:

+------------+-----------------------------------------------------------------------------------+
| Rank type  | Description                                                                       |
+============+===================================================================================+
| ``'ofv'``  | ΔOFV. Default is to not rank candidates with ΔOFV < cutoff (default 3.84)         |
+------------+-----------------------------------------------------------------------------------+
| ``'aic'``  | ΔAIC. Default is to rank all candidates if no cutoff is provided.                 |
+------------+-----------------------------------------------------------------------------------+
| ``'bic'``  | ΔBIC (mixed). Default is to rank all candidates if no cutoff is provided.         |
+------------+-----------------------------------------------------------------------------------+

Information about how BIC is calculated can be found in :py:func:`pharmpy.modeling.calculate_bic`.

~~~~~~~~~~~~~~~~~~~~~~~
The Modelsearch results
~~~~~~~~~~~~~~~~~~~~~~~

The results object contains various summary tables which can be accessed in the results object, as well as files in
.csv/.json format. The name of the selected best model (based on the input selection criteria) is also included.

Consider a modelsearch run with the search space of zero and first order absorption, adding zero or one peripheral
compartment and lagtime:

.. pharmpy-code::

    res = run_modelsearch(search_space='ABSORPTION([FO,ZO]);PERIPHERALS([0,1]);LAGTIME(ON)',
                          algorithm='reduced_stepwise',
                          model=start_model,
                          results=start_model_results,
                          iiv_strategy='absorption_delay',
                          rank_type='bic',
                          cutoff=None)

The ``summary_tool`` table contains information such as which feature each model candidate has, the difference to the
start model (in this case comparing BIC), and final ranking:

.. pharmpy-execute::
    :hide-code:

    from pharmpy.workflows.results import read_results
    res = read_results('tests/testdata/results/modelsearch_results.json')
    res.summary_tool

To see information about the actual model runs, such as minimization status, estimation time, and parameter estimates,
you can look at the ``summary_models`` table. The table is generated with
:py:func:`pharmpy.tools.summarize_modelfit_results`.

.. pharmpy-execute::
    :hide-code:

    res.summary_models

Finally, you can see a summary of different errors and warnings in ``summary_errors``.
See :py:func:`pharmpy.tools.summarize_errors` for information on the content of this table.

.. pharmpy-execute::
    :hide-code:

    import pandas as pd
    pd.set_option('display.max_colwidth', None)
    res.summary_errors
