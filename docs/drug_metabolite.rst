.. _drug_metabolite:

===============
Drug metabolite
=============== 

~~~~~~~
Running
~~~~~~~

~~~~~~
Models
~~~~~~

Currently implemented drug metabolite models are:

* Basic metabolite

    * Single metabolite compartment with parent -> metabolite conversion of 100%

* Basic metabolite with a (metabolite) peripheral compartment(s)

* Presystemic drug metabolite (PSC)

    * Presystemic metabolite compartment with parent -> metabolite conversion of 100%

* Presystemic drug metabolite with a (metabolite) peripheral compartment(s)

~~~~~~~~~~~~~~~~~~~~~
Structsearch workflow
~~~~~~~~~~~~~~~~~~~~~

The graph below show how the drug metabolite models are built, with each of the two types 
of drug metabolite models (basic and pre-systemic) with and without added peripherals.
For reasons explained above, one of the created candidate models will be chosen as 
the base model (shown by a square). The base model is chosen as one of the candidate models
with the fewewst amount of peripheral compartments as possible, with "BASIC" being chosen over
"PSC". If the inputed model is a drug_metabolite model, this will be used as the base model 
instead.

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
    Peripheral compartments are added using the see :ref:`exhaustive stepwise search algorithm<algorithms_modelsearch>`.

Regarding DVID, DVID=1 is connected to the parent drug while DVID=2 is representing the metabolite.

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
| PERIPHERALS    | `number`, MET                | Regular PERIPHERALS with second option set to MET                  |
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

~~~~~~~
Results
~~~~~~~
