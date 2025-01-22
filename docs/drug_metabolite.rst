.. _drug_metabolite:

===============
Drug metabolite
=============== 

~~~~~~~
Running
~~~~~~~

The code to initiate structsearch for a drug metabolite model in Python/R is stated below:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools read_modelfit_results, run_structsearch

    start_model = read_model('path/to/model')
    start_model_results = read_modelfit_results('path/to/model')
    res = run_structsearch(model=start_model,
                           results=start_model_results,
                           type='drug_metabolite',
                           search_space="METABOLITE([BASIC,PSC]);PERIPHERALS(0..1,MET)")

.. note::                          
    The tool expect the model to have a dataset with two DVIDs. DVID=1 connected to the parent
    drug and DVID=2 which is representing the metabolite.

+-------------------------------------------------+-----------------------------------------------------------------------+
| Argument                                        | Description                                                           |
+=================================================+=======================================================================+
| ``model``                                       | Start model                                                           |
+-------------------------------------------------+-----------------------------------------------------------------------+
| ``results``                                     | :code:`ModelfitResults` object of the start model                     |
+-------------------------------------------------+-----------------------------------------------------------------------+
| ``type``                                        | Need to be set to 'drug_metabolite' (see :ref:`type<the model types>` |
|                                                 | for more)                                                             |
+-------------------------------------------------+-----------------------------------------------------------------------+
| ``search_space``                                | :ref:`Search space<drug metabolite search space>` of models to test   |
+-------------------------------------------------+-----------------------------------------------------------------------+
| ``strictness``                                  | :ref:`Strictness<strictness>` criteria for model selection.           |
|                                                 | Default is :code:`"minimization_successful or                         |
|                                                 | (rounding_errors and sigdigs>= 0.1)"`                                 |
+-------------------------------------------------+-----------------------------------------------------------------------+


~~~~~~
Models
~~~~~~

Currently implemented drug metabolite models are:

+--------------------------------+----------------------------------------------------------+
| Model type                     | Description                                              |
+--------------------------------+----------------------------------------------------------+
| Basic metabolite               | Single metabolite compartment with parent -> metabolite  |
|                                | conversion of 100%.                                      |
+--------------------------------+----------------------------------------------------------+
| Basic metabolite with          | Same as 'Basic metabolite' with one or more connected    |
| peripheral compartment(s)      | peripheral compartments.                                 |
+--------------------------------+----------------------------------------------------------+
| Presystemic drug metabolite    | Presystemic metabolite compartment with parent ->        |
| (PSC)                          | metabolite conversion of 100%.                           |
+--------------------------------+----------------------------------------------------------+
| Presystemic drug metabolite    | Same as 'PSC' with one or more connected peripheral      |
| with peripheral compartment(s) | connected compartments.                                  |
+--------------------------------+----------------------------------------------------------+

.. note::
    Drug metabolite models created outside of pharmpy currently require the metabolite
    compartment to be named 'METABOLITE' in order to function properly.


.. _drug metabolite search space:

~~~~~~~~~~~~~~~~
The search space
~~~~~~~~~~~~~~~~

MFL support the following model features:

+---------------+-------------------------------+--------------------------------------------------------------------+
| Category      | Options                       | Description                                                        |
+===============+===============================+====================================================================+
| METABOLITE    | :code:`PSC, BASIC`            | Type of drug metabolite model to add. PSC is for presystemic       |
+---------------+-------------------------------+--------------------------------------------------------------------+
| PERIPHERALS   | :code:`number, MET`           | Regular PERIPHERALS with second option set to MET                  |
+---------------+-------------------------------+--------------------------------------------------------------------+

A search space for testing both BASIC and PSC (presystemic) drug metabolite models with 0 or 1 peripheral compartments 
for the metabolite compartment would look like:
.. code-block::

    METABOLITE([BASIC,PSC]);PERIPHERALS(0..1,MET)


This can be combined with the search space for the modelsearch tool by simply adding the drug metabolite features to it.
Please see the example below. Note that two peripherals statements are present, one for the drug and one for the metabolite.

.. code-block::

    ABSORPTION(FO);ELIMINATION(FO);PERIPHERALS(0,1);METABOLITE(PSC);PERIPHERAL(0..1,MET)

When running through AMD, if a search space is not specified, a default one will be taken based on the administration type.

If administration is oral or ivoral, the search space will be as follows:

.. code-block::

    METABOLITE([BASIC,PSC]);PERIPHERALS(0..1,MET)

But with an iv administration instead, the default search space becomes:

.. code-block::

    METABOLITE(BASIC);PERIPHERALS(0..1,MET)

.. _results:

~~~~~~~~~~~~~~~~~~~~~
Structsearch workflow
~~~~~~~~~~~~~~~~~~~~~

The given input model can either be a drug metabolite model or a pk model.

.. note::
    With a drug metabolite model as input, only PERIPHERALS are supported in the search space.

The graph below shows how the drug metabolite models are built, with each of the two types 
of basic and pre-systemic, with and without added peripherals. If the given input model is a
drug metabolite model, this will be used as the base model. If not, the base model is chosen
as the candidate models with the fewest amount of peripheral compartments as possible, with
"BASIC" being chosen over "PSC".

.. graphviz::

    digraph BST {
            node [fontname="Arial"]
            base [label="Base model"]
            s1 [label="Base metabolite";shape = rect;]
            s2 [label="PERIPHERALS(0)"]
            s3 [label="PERIPHERALS(1)"]
            s4 [label="Presystemic metabolite"]
            s5 [label="PERIPHERALS(0)"]
            s6 [label="PERIPHERALS(1)"]

            base -> s1
            s1 -> s2
            s1 -> s3
            base -> s4
            s4 -> s5
            s4 -> s6
    }

.. note::
    Peripheral compartments are added using the :ref:`exhaustive stepwise search algorithm<algorithms_modelsearch>`.

~~~~~~~
Results
~~~~~~~

The results object contains various summary tables which can be accessed in the results object, as well as files in
.csv/.json format. The name of the selected best model (based on the input selection criteria) is also included.

Below is an example for a drug metabolite run.

.. pharmpy-code::

    res = run_structsearch(model=start_model,
                           results=start_model_results,
                           type='drug_metabolite',
                           search_space="METABOLITE([BASIC,PSC]);PERIPHERALS(0..1,MET)")

The ``summary_tool`` table contains information such as which feature each model candidate has, the difference to the
start model (in this case comparing BIC), and final ranking:

.. pharmpy-execute::
   :hide-code:

    from pharmpy.workflows.results import read_results
    res = read_results('tests/testdata/results/structsearch_results_drug_metabolite.json')
    res.summary_tool
