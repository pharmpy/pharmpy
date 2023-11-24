.. _structsearch:

============
Structsearch
============

The structsearch tool is a tool to find the best structural model given a base model and a specific model scope to test. 
This tool will always return a model of the specified type.
Currently supported model types are PKPD, TMDD and drug metabolite.


.. toctree::
   :maxdepth: 1

   PKPD <pkpd>
   TMDD <tmdd>
   Drug Metabolite <drug_metabolite>

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


Arguments
~~~~~~~~~
The arguments of the structsearch tool are listed below.

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
| ``extra_model``                                 | Extra model for TMDD structsearch (only for TMDD)                   |
+-------------------------------------------------+---------------------------------------------------------------------+
| ``extra_model_results``                         | ModelfitResults object for the extra model for TMDD structsearch    |
|                                                 | (TMDD only)                                                         |
+-------------------------------------------------+---------------------------------------------------------------------+

.. _the model types:

~~~~~~~~~~~~~~~
The model types
~~~~~~~~~~~~~~~

Structsearch is currently available for PKPD and drug metabolite models.
For mode detailed information about the model types see the references.

+--------------------------+--------------------------------------------+
| type                     | Description                                |
+==========================+============================================+
| :code:`pkpd`             | PKPD models (see :ref:`pkpd`)              |
+--------------------------+--------------------------------------------+
| :code:`drug_metabolite`  | Drug metabolite models                     |
+--------------------------+--------------------------------------------+
| :code:`TMDD`             | TMDD models (see :ref:`tmdd`)              |
+--------------------------+--------------------------------------------+


When creating candidate models for the specified model type, some candidate models will be derived from the base model 
and others from another candidate model. When computing the results, each candidate models' 
results is compared to that of its parent. The candidate models should not be compared to the base model due to them  being 
of different model types. For this reason, the least complex candidate model is chosen as the default model for this tool. 
This model replaces the parent of all candidate models which have the base model as its parent. In the graphs below, this model
is represented by a rectangel. This ensures correct comparison of model results as well as establishing that a model of the 
speciefied type will always be returned.


.. _the search space:

~~~~~~~~~~~~~~~~
The search space
~~~~~~~~~~~~~~~~

The model feature search space is a set of possible combinations of model features that will be applied and tested on
the input model. The supported features cover absorption, absorption delay, elimination, and distribution. The search
space is given as a string with a specific grammar, according to the `Model Feature Language` (MFL) (see :ref:`detailed description<mfl>`).

For detailed information about the search space for the different model types please see the respective page.


~~~~~~~
Results
~~~~~~~

The results object contains various summary tables which can be accessed in the results object, as well as files in
.csv/.json format. The name of the selected best model (based on the input selection criteria) is also included.

Example results for the different model types can be found on the respective pages.
