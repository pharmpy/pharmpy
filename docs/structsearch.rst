.. _structsearch:

============
Structsearch
============

The structsearch tool is a tool to find the best structural model given a base model and a specific model scope to test. 
This tool will always return a model of the specified type.
Currently, supported model types are PKPD and drug metabolite.


~~~~~~~
Running
~~~~~~~

The structsearch tool is available both in Pharmpy/pharmr.

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
.. note::
    For drug_metabolite models, the argument 'route' need to be set according to the table below.


Arguments
~~~~~~~~~
The arguments of the structsearch tool are listed below.

+-------------------------------------------------+---------------------------------------------------------------------+
| Argument                                        | Description                                                         |
+=================================================+=====================================================================+
| :ref:`type<the model types>`                    | Type of model. Can be either pkpd or drug_metabolite                |
+-------------------------------------------------+---------------------------------------------------------------------+
| route                                           | Type of administration. Currently 'oral', 'iv' and 'ivoral'         |
|                                                 | (currently only for drug_metabolite models)                         |
+-------------------------------------------------+---------------------------------------------------------------------+
| :ref:`search_space<the search space>`           | Search space of models to test (currently only for PKPD models)     |
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
|                                                 | (rounding_errors and sigdigs>= 0)"                                  |
+-------------------------------------------------+---------------------------------------------------------------------+
| ``extra_model``                                 | Extra model for TMDD structsearch (only for TMDD)                   |
+-------------------------------------------------+---------------------------------------------------------------------+

.. _the model types:

~~~~~~~~~~~~~~~
The model types
~~~~~~~~~~~~~~~

Structsearch is currently available for PKPD and drug metabolite models.

+--------------------------+--------------------------------------------+
| type                     | Description                                |
+==========================+============================================+
| :code:`pkpd`             | PKPD models                                |
+--------------------------+--------------------------------------------+
| :code:`drug_metabolite`  | Drug metabolite models                     |
+--------------------------+--------------------------------------------+
| :code:`TMDD`             | TMDD models                                |
+--------------------------+--------------------------------------------+

When creating candidate models for the specified model type, some candidate models will be derived from the base model 
and others from another candidate model. When computing the results, each candidate models' 
results is compared to that of its parent. The candidate models should not be compared to the base model due to them  being 
of different model types. For this reason, the least complex candidate model is chosen as the default model for this tool. 
This model replaces the parent of all candidate models which have the base model as its parent. In the graphs below, this model
is represented by a rectangel. This ensures correct comparison of model results as well as establishing that a model of the 
speciefied type will always be returned.


PKPD models
~~~~~~~~~~~

Currently implemented PKPD models are: 

* :py:func:`Direct effect models<pharmpy.modeling.set_direct_effect>`.

* :py:func:`Effect compartment models<pharmpy.modeling.add_effect_compartment>`.

* :py:func:`Indirect effect models<pharmpy.modeling.add_indirect_effect>`.

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


Drug metabolite
~~~~~~~~~~~~~~~

Currently implemented drug metabolite models are:

* Base metabolite

    * Single metabolite compartment with parent -> metabolite conversion of 100%

* Base metabolite with a (metabolite) peripheral compartment

* Presystemic drug metabolite

    * Presystemic metabolite compartment with parent -> metabolite conversion of 100%

* Presystemic drug metabolite with a (metabolite) peripheral compartment

.. note::
    Presystemic drug metabolite models are only created if route is set to 'oral' or 'ivoral'

.. graphviz::

    digraph BST {
            node [fontname="Arial"]
            base [label="Base model"]
            s1 [label="Base metabolite";shape = rect;]
            s2 [label="Base metabolite with peripheral"]
            s3 [label="Presystemic metabolite"]
            s4 [label="Presystemic metabolite with peripheral"]

            base -> s1
            s1 -> s2
            base -> s3
            s3 -> s4
    }

Regarding DVID, DVID=1 is connected to the parent drug while DVID=2 is representing the metabolite.


TMDD models
~~~~~~~~~~~

Implemented target mediated drug disposition (TMDD) models are:

- Full model
- Irreversible binding approximation (IB)
- Constant total receptor approximation (CR)
- Irreversible binding and constant total receptor approximation (CR+IB)
- Quasi steady-state approximation (QSS)
- Wagner
- Michaelis-Menten approximation (MMAPP)

The structsearch procedure is as follows:

1. Perform modelsearch
2. Get the final model of the modelsearch and a model with the same features as the final model but with one
   less peripheral compartment if one such model exists.
3. Create 8 QSS models for the final model and 8 QSS models for the final model minus one compartment if it exists.
   Otherwise only 8 QSS models are created.
4. Find best QSS model of the 16(8) QSS models
5. Create 4 full models, 2 CR+IB models, 1 Wagner model, 2 CR models,
   2 IB models and 1 MMAPP model. Use parameter estimates from the best QSS model as initial estimates for the
   generated models.
6. Find the best model of these 12 models.


.. graphviz::

    digraph BST {
            node [fontname="Arial"];
            base [label="Base model"]
            s0 [label="Modelsearch"]
            s1 [label="final model (+ final model -1 comp)"]
            s2 [label="8 (+ 8) QSS models"]
            s3 [label="best QSS model"]
            s31 [label="4 full"]
            s32 [label="2 CR+IB"]
            s33 [label="1 Wagner"]
            s34 [label="2 CR"]
            s35 [label="2 IB"]
            s36 [label="1 MMAPP"]

            base -> s0
            s0 -> s1
            s1 -> s2
            s2 -> s3
            s3 -> s31
            s3 -> s32
            s3 -> s33
            s3 -> s34
            s3 -> s35
            s3 -> s36
    }


.. note::

    Please note that only steps 3-6 are performed inside the structsearch tool. The structsearch tool takes two models
    as input arguments and creates the 16 QSS models from them. 
    Steps 1 and 2 are performed outside of the structsearch tool. These steps are implemented in the AMD tool but can
    alternatively be created by the user.

.. _the search space:

~~~~~~~~~~~~~~~~
The search space
~~~~~~~~~~~~~~~~

The model feature search space is a set of possible combinations of model features that will be applied and tested on
the input model. The supported features cover absorption, absorption delay, elimination, and distribution. The search
space is given as a string with a specific grammar, according to the `Model Feature Language` (MFL) (see :ref:`detailed description<mfl>`).

.. note::
    At the moment a search space is only defined for PKPD models.


PKPD
~~~~

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
check :ref:`model types<the model types>`.

To test all direct effect models the search space looks as follows:


.. code-block::

    DIRECTEFFECT(*)


Search space for testing linear and emax models for direct effect and effect compartment models:

.. code-block::

    DIRECTEFFECT([linear, emax])
    EFFECTCOMP([linear, emax])


.. _the structsearch results:


~~~~~~~~~~~~~~~~~~~~~~~~
The Structsearch results
~~~~~~~~~~~~~~~~~~~~~~~~

The results object contains various summary tables which can be accessed in the results object, as well as files in
.csv/.json format. The name of the selected best model (based on the input selection criteria) is also included.

Below is an example for a PKPD run. Drug metabolite results follow the same structure.

.. pharmpy-code::

    res = run_structsearch(type='pkpd',
                            search_space=DIRECTEFFECT(emax);EFFECTCOMP([linear,emax])",
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
