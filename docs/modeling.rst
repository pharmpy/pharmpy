========
Modeling
========

While the :py:class:`pharmpy.model.Model` class can be directly manipulated
with low level operations the modeling module offers higher level operations and transformations for building a model.
These transformations are also available via the Pharmpy command line interface.

.. jupyter-execute::
   :hide-output:
   :hide-code:

   from pathlib import Path
   path = Path('tests/testdata/nonmem/')
   from docs.help_functions import print_model_diff

The following model is the start model for the examples.

.. jupyter-execute::

   from pharmpy import Model

   model_ref = Model(path / "pheno.mod")
   print(model_ref)


~~~~~~~~~~~~~~~~~~~~~~~~~
PK models and ODE systems
~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   model = Model(path / "pheno.mod")

The ODE system of a PK model can be converted from having a compartmental description to be described with an explicit ODE-system.

.. jupyter-execute::

   from pharmpy.modeling import explicit_odes

   print(model.statements.ode_system)
   explicit_odes(model)
   print(model.statements.ode_system)

For NONMEM models this means going from any of the compartmental ADVANS (ADVAN1-4, ADVAN10-12) to coding using an explicit $DES.

.. jupyter-execute::

   model.update_source()
   print_model_diff(model_ref, model)

Absorption rate
~~~~~~~~~~~~~~~

The :py:func:`pharmpy.modeling.absorption_rate` can be used to set the absorption rate.


Bolus absorption
==================

Let us use a model with bolus absorption as a starting point.

.. graphviz::

   digraph fo {
     rankdir = LR
     node [shape=box]
     S [label="S", style=invis, width=0, height=0, margin=0];
     Output [label="O", style=invis, width=0, height=0, margin=0];
     "Central" -> Output [label=K];
     S -> "Central" [label="Bolus"];
   }

.. jupyter-execute::

   from pharmpy.modeling import absorption_rate
   model = Model(path / "pheno.mod")

This type of absorption can be created with

.. jupyter-execute::

    absorption_rate(model, 'bolus')
    model.update_source()
    print_model_diff(model_ref, model)


Zero order
===========

Let us now change to zero order absorption.

.. graphviz::

   digraph fo {
     rankdir = LR
     node [shape=box]
     S [label="S", style=invis, width=0, height=0, margin=0];
     Output [label="O", style=invis, width=0, height=0, margin=0];
     "Central" -> Output [label=K];
     S -> "Central" [label=Infusion];
   }

.. jupyter-execute::

   absorption_rate(model, 'ZO')
   model.update_source(nofiles=True)
   print_model_diff(model_ref, model)

First order
===========

First order absorption would mean adding an absorption (depot) compartment like this

.. graphviz::

   digraph fo {
     rankdir = LR
     node [shape=box]
     S [label="S", style=invis, width=0, height=0, margin=0];
     Output [label="O", style=invis, width=0, height=0, margin=0];
     "Depot" -> "Central" [label=Ka];
     "Central" -> Output [label=K];
     S -> "Depot" [label=Bolus];
   }

.. jupyter-execute::

   absorption_rate(model, 'FO')
   model.update_source(nofiles=True)
   print_model_diff(model_ref, model)

Sequential zero-order then first-order
======================================

Sequential zero-order absorption followed by first-order absorption will have an infusion dose into the depot compartment

.. graphviz::

   digraph fo {
     rankdir = LR
     node [shape=box]
     S [label="S", style=invis, width=0, height=0, margin=0];
     Output [label="O", style=invis, width=0, height=0, margin=0];
     "Depot" -> "Central" [label=Ka];
     "Central" -> Output [label=K];
     S -> "Depot" [label=Infusion];
   }

.. jupyter-execute::

   absorption_rate(model, 'seq-ZO-FO')
   model.update_source(nofiles=True)
   print_model_diff(model_ref, model)


~~~~~~~~~~~~~~~~~~~~~~~~~
Adding covariate effects
~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   model = Model(path / "pheno.mod")

Covariate effects may also be applied to a model.

.. jupyter-execute::
   :hide-output:

   from pharmpy.modeling import add_covariate_effect
   add_covariate_effect(model, 'CL', 'WGT', 'pow')

Here, *CL* indicates the name of the parameter onto which you want to apply the effect, *WGT* is the covariate, and
*pow* (power function) is the effect you want to apply.
See :py:class:`pharmpy.modeling.add_covariate_effect` for effects with available templates.

.. jupyter-execute::

   model.update_source()
   print_model_diff(model_ref, model)

Pharmpy also supports user formatted covariate effects.

.. jupyter-execute::
   :hide-output:

   model = Model(path / "pheno.mod")
   user_effect = '((cov/std) - median) * theta'
   add_covariate_effect(model, 'CL', 'WGT', user_effect)

It is necessary that the names follow the same format as in user_effect, meaning that the covariate is denoted as
*cov*, the theta as *theta* (or, if multiple thetas: *theta1*, *theta2* etc.), and the mean or median as *mean* and *median*, respectively. This is in order for
the names to be substituted with the correct values.

.. jupyter-execute::

   model.update_source()
   print_model_diff(model_ref, model)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Transformation of etas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Boxcox
~~~~~~

.. jupyter-execute::

   model = Model(path / "pheno.mod")

To apply a boxcox transformation, input a list of the etas of interest.

.. jupyter-execute::

   from pharmpy.modeling import boxcox
   boxcox(model, ['ETA(1)'])
   model.update_source()
   print_model_diff(model_ref, model)

This can be done for one or multiple etas. The new model will have new statements where *ETAB1* is a boxcox
transformation of *ETA(1)*.

If no list is provided, all etas will be updated.

.. jupyter-execute::

   model = Model(path / "pheno.mod")
   boxcox(model)
   model.update_source()
   print_model_diff(model_ref, model)

Approximate t-distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~

Applying an approximate t-distribution transformation of etas is analogous to a boxcox transformation. The input
is similarly a list of etas, and if no list is provided all etas will be transformed.

.. jupyter-execute::

   model = Model(path / "pheno.mod")
   from pharmpy.modeling import tdist
   tdist(model, ['ETA(1)'])
   model.update_source()
   print_model_diff(model_ref, model)

John Draper
~~~~~~~~~~~

Similarly, a John Draper transformation uses a list of etas as input, if no list is
provided all etas will be transformed.

.. jupyter-execute::

   model = Model(path / "pheno.mod")
   from pharmpy.modeling import john_draper
   john_draper(model, ['ETA(1)'])
   model.update_source()
   print_model_diff(model_ref, model)

