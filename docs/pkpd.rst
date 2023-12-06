.. _pkpd:

====
PKPD
====

~~~~~~~
Running
~~~~~~~

The code to initiate structsearch for a PKPD model in Python/R is stated below:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools read_modelfit_results, run_structsearch

    start_model = read_model('path/to/model')
    start_model_results = read_modelfit_results('path/to/model')
    res = run_structsearch(type='pkpd',
                            search_space="DIRECTEFFECT(*)",
                            model=start_model,
                            results=start_model_results)


This will take an input model ``model`` with a ``search_space`` that includes all direct effect PKPD models.

.. note::
    For PKPD models the input model has to be a PK model with a PKPD dataset. 


Arguments
~~~~~~~~~
The arguments of the structsearch tool for PKPD models are listed below.

+-------------------------------------------------+---------------------------------------------------------------------+
| Argument                                        | Description                                                         |
+=================================================+=====================================================================+
| :ref:`type<the model types>`                    | Type of model. Can be either pkpd or drug_metabolite                |
+-------------------------------------------------+---------------------------------------------------------------------+
| :ref:`search_space<the search space>`           | Search space of models to test                                      |
+-------------------------------------------------+---------------------------------------------------------------------+
| b_init (optional, default is 0.1)               | Initial estimate for baseline effect (only for PKPD models)         |
+-------------------------------------------------+---------------------------------------------------------------------+
| emax_init (optional, default is 0.1)            | Initial estimate for E :sub:`max` parameter (only for PKPD models)  |
+-------------------------------------------------+---------------------------------------------------------------------+
| ec50_init (optional, default is 0.1)            | Initial estimate for EC :sub:`50` parameter (only for PKPD models)  |
+-------------------------------------------------+---------------------------------------------------------------------+
| met_init (optional, default is 0.1)             | Initial estimate for mean equilibration time  (only for PKPD models)|
+-------------------------------------------------+---------------------------------------------------------------------+
| ``model``                                       | PK start model                                                      |
+-------------------------------------------------+---------------------------------------------------------------------+
| ``results``                                     | ModelfitResults of the start model                                  |
+-------------------------------------------------+---------------------------------------------------------------------+
| :ref:`strictness<strictness>`                   | Strictness criteria for model selection.                            |
|                                                 | Default is "minimization_successful or                              |
|                                                 | (rounding_errors and sigdigs>= 0.1)"                                |
+-------------------------------------------------+---------------------------------------------------------------------+

.. _models:

~~~~~~
Models
~~~~~~

Currently implemented PKPD models are: 

* :py:func:`Direct effect models<pharmpy.modeling.set_direct_effect>`.

* :py:func:`Effect compartment models<pharmpy.modeling.add_effect_compartment>`.

* :py:func:`Indirect effect models<pharmpy.modeling.add_indirect_effect>`.


~~~~~~~~~~~~~~~~~~~~~
Structsearch workflow
~~~~~~~~~~~~~~~~~~~~~

PKPD candidate models will be derived from the input PK model. Additionaly to the candidate models a baseline
model is created, which serves as a reference model when the models are ranked. 

.. graphviz::

    digraph BST {
            node [fontname="Arial"]
            base [label="Base model"]
            s1 [label="Baseline";shape = rect;]
            s2 [label="direct effect linear"]
            s3 [label="direct effect emax"]
            s4 [label="direct effect sigmoid"]
            s5 [label="effect compartment linear"]
            s6 [label="..."]

            base -> s1
            base -> s2
            base -> s3
            base -> s4
            base -> s5
            base -> s6
    }

Note : The figure above is only showing a subset of all candidate models created, indicated by "..."

Regarding DVID, DVID=1 is representing PK observations while DVID=2 is connected to PD observations.

.. _the search space:

~~~~~~~~~~~~
Search space
~~~~~~~~~~~~ 

MFL support the following model features:

+---------------+-------------------------------+--------------------------------------------------------------------+
| Category      | Options                       | Description                                                        |
+===============+===============================+====================================================================+
| DIRECTEFFECT  | `model`                       | Direct effect PD models.                                           |
+---------------+-------------------------------+--------------------------------------------------------------------+
| EFFECTCOMP    | `model`                       | Effect comprtment PD models.                                       |
+---------------+-------------------------------+--------------------------------------------------------------------+
| INDIRECTEFFECT| `model`, `option`             | Indirect effect PD models. `option` can be                         |
|               |                               | either production or degradation.                                  |
+---------------+-------------------------------+--------------------------------------------------------------------+

The option `model` describes a PKPD model, such as E :sub:`max`. For more details
check :ref:`model types<models>`.

To test all direct effect models the search space looks as follows:


.. code-block::

    DIRECTEFFECT(*)


Search space for testing linear and emax models for direct effect and effect compartment models:

.. code-block::

    DIRECTEFFECT([linear, emax])
    EFFECTCOMP([linear, emax])

~~~~~~~
Results
~~~~~~~

The results object contains various summary tables which can be accessed in the results object, as well as files in
.csv/.json format. The name of the selected best model (based on the input selection criteria) is also included.

Below is an example for a PKPD run.

.. pharmpy-code::

    res = run_structsearch(type='pkpd',
                            search_space="DIRECTEFFECT(emax);EFFECTCOMP([linear,emax])",
                            model=start_model,
                            results=start_model_results)

The ``summary_tool`` table contains information such as which feature each model candidate has, the difference to the
start model (in this case comparing BIC), and final ranking:

.. pharmpy-execute::
   :hide-code:

    from pharmpy.workflows.results import read_results
    res = read_results('tests/testdata/results/structsearch_results_pkpd.json')
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