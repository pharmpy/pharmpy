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
                            b_init = 0.1,
                            emax_init = 0.1,
                            ec50_init = 0.7,
                            met_init = 0.3,
                            results=start_model_results)


This will take an input model ``model`` with a ``search_space`` that includes all direct effect PKPD models.

.. note::
    For PKPD models the input model has to be a PK model with a PKPD dataset. 


Arguments
~~~~~~~~~
The arguments of the structsearch tool for PKPD models are listed below.

Mandatory
---------

+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| Argument                                        | Description                                                                             |
+=================================================+=========================================================================================+
| ``type``                                        | Type of model. In this case "pkpd".                                                     |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``model``                                       | PK start model                                                                          |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``results``                                     | ModelfitResults of the start model                                                      |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+

Optional
--------

+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| Argument                                        | Description                                                                             |
+=================================================+=========================================================================================+
| ``b_init``                                      | Initial estimate for baseline effect. Optional. Default is 0.1                          |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``emax_init``                                   | Initial estimate for E :sub:`max` parameter.                                            |
|                                                 | Default is 0.1                                                                          |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``ec50_init``                                   | Initial estimate for EC :sub:`50` parameter.                                            |
|                                                 | Default is 0.1                                                                          |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``met_init``                                    | Initial estimate for mean equilibration time.                                           |
|                                                 | Default is 0.1                                                                          |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``search_space``                                | :ref:`Search space<the search space pkpd>` of models to test. Optional.                 |
|                                                 | If ``None`` all implemented models are used.                                            |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``strictness``                                  | :ref:`Strictness<strictness>` criteria for model selection.                             |
|                                                 | Default is "minimization_successful or                                                  |
|                                                 | (rounding_errors and sigdigs>= 0.1)".                                                   |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+

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

.. _the search space pkpd:

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

Search space for testing linear indirect effect degradation models:

.. code-block::

    INDIRECTEFFECT(linear,DEGRADATION)

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

~~~~~~~~
Examples
~~~~~~~~

Minimum required arguments to run structsearch for PKPD models:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools read_modelfit_results, run_structsearch

    start_model = read_model('path/to/model')
    start_model_results = read_modelfit_results('path/to/model')

    res = run_structsearch(type='pkpd',
                            model=start_model,
                            results=start_model_results)

Specifying initial parameters:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools read_modelfit_results, run_structsearch

    start_model = read_model('path/to/model')
    start_model_results = read_modelfit_results('path/to/model')

    res = run_structsearch(type='pkpd',
                            model=start_model,
                            results=start_model_results,
                            b_init = 0.09, e_max_init = 3, ec50_init = 1.5)


Run structsearch with initial estimates for all direct effect models and all indirect effect models with production:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools read_modelfit_results, run_structsearch

    start_model = read_model('path/to/model')
    start_model_results = read_modelfit_results('path/to/model')

    res = run_structsearch(type='pkpd',
                            model=start_model,
                            results=start_model_results,
                            b_init = 0.09, e_max_init = 3, ec50_init = 1.5,
                            search_space = "DIRECTEFFECT(*);INDIRECTEFFECT(*,PRODUCTION)")
