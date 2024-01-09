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
Which arguments are mandatory or optional depends on the type of model. More information can be found on the respective page.

+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| Argument                                        | Description                                                                             |
+=================================================+=========================================================================================+
| ``type``                                        | Type of model. Can be either pkpd, drug_metabolite or tmdd (:ref:`the model types`).    |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``search_space``                                | :ref:`Search space<structsearch search space>` of models to test.                       |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``b_init``                                      | Initial estimate for baseline effect (only for PKPD models). Default is 0.1.            |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``emax_init``                                   | Initial estimate for E :sub:`max` parameter (only for PKPD models).                     |
|                                                 | Default is 0.1.                                                                         |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``ec50_init``                                   | Initial estimate for EC :sub:`50` parameter (only for PKPD models).                     |
|                                                 | Default is 0.1.                                                                         |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``met_init``                                    | Initial estimate for mean equilibration time  (only for PKPD models).                   |
|                                                 | Default is 0.1.                                                                         |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``model``                                       | PK start model.                                                                         |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``results``                                     | ModelfitResults of the start model.                                                     |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``strictness``                                  | :ref:`Strictness<strictness>` criteria for model selection. Optional.                   |
|                                                 | Default is "minimization_successful or                                                  |
|                                                 | (rounding_errors and sigdigs>= 0.1)".                                                   |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``extra_model``                                 | Extra model for TMDD structsearch (only for TMDD). Optional.                            |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``extra_model_results``                         | ModelfitResults object for the extra model for TMDD structsearch                        |
|                                                 | (TMDD only). Optional.                                                                  |
+-------------------------------------------------+-----------------------------------------------------------------------------------------+

.. _the model types:

~~~~~~~~~~~~~~~
The model types
~~~~~~~~~~~~~~~

Structsearch is currently available for PKPD, drug metabolite and TMDD models.
For more detailed information about the model types see the references.

+--------------------------+---------------------------------------------------------+
| type                     | Description                                             |
+==========================+=========================================================+
| :code:`pkpd`             | :ref:`PKPD models<pkpd>`                                |
+--------------------------+---------------------------------------------------------+
| :code:`drug_metabolite`  | :ref:`Drug metabolite models<drug_metabolite>`          |
+--------------------------+---------------------------------------------------------+
| :code:`TMDD`             | :ref:`TMDD models<tmdd>`                                |
+--------------------------+---------------------------------------------------------+

.. _structsearch search space:

~~~~~~~~~~~~~~~~
The search space
~~~~~~~~~~~~~~~~

The search space is a set of possible model types that are allowed for the final model (e.g. only direct effect models 
for PKPD models). 
The search space is given as a string with a specific grammar, according to the `Model Feature Language` (MFL) 
(see :ref:`detailed description<mfl>`).

For detailed information about the search space for the different model types in structsearch please see
the respective page.


~~~~~~~
Results
~~~~~~~

The results object contains various summary tables which can be accessed in the results object, as well as files in
.csv/.json format. The name of the selected best model (based on the input selection criteria) is also included.

Example results for the different model types can be found on the respective pages.
