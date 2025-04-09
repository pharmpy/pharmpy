.. _mfl:

============================
Model feature language (MFL)
============================

The `MFL` is a domain specific language designed to describe model features and sets of model features in a concise way.
It can be used to describe model features for one single model or an entire space of model features, i.e. descriptions
for multiple models. The basic building block of MFL is the feature description. A feature description consists of the
name of a feature category followed by a comma separated list of arguments within parentheses. For example:

.. code::

    ABSORPTION(FO)

Each feature description describes one or multiple features of the same category, i.e. absorption, absorption delay,
elimination, and distribution. Features of the same category are mutually exclusive and cannot be applied to the same
model. Multiple model feature descriptions can be combined by separating them with either newline or a semi-colon.

The following two examples are equivalent:

.. code::

    ABSORPTION(FO);ELIMINATION(ZO)

.. code::

    ABSORPTION(FO)
    ELIMINATION(ZO)

Option types
============

MFL support the following types of options to feature descriptions:

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
| symbol        | :code:`@IIV`     | A symbol with manual or automatic definition          |
+---------------+------------------+-------------------------------------------------------+

Model features
==============

MFL support the following model features:

+---------------+--------------------------------+--------------------------------------------------------------------+
| Category      | Options                        | Description                                                        |
+===============+================================+====================================================================+
| ABSORPTION    | :code:`INST, FO, ZO, SEQ-ZO-FO`| Absorption rate (instantaneous, first order, zero order, sequential|
|               |                                | zero order first order)                                            |
+---------------+--------------------------------+--------------------------------------------------------------------+
| ELIMINATION   | :code:`FO, ZO, MM, MIX-FO-MM`  | Elimination rate (first order, zero order, Michaelis-Menten,       |
|               |                                | mixed first order Michaelis-Menten)                                |
+---------------+--------------------------------+--------------------------------------------------------------------+
| PERIPHERALS   | `number`, DRUG/MET             | Number of peripheral compartments. If the peripherals compartment  |
|               |                                | should be added for the drug compartment (default) or to the       |
|               |                                | metabolite compartment (if any). Only applicable to drug_metabolite|
|               |                                | models.                                                            |
+---------------+--------------------------------+--------------------------------------------------------------------+
| TRANSITS      | `number`, DEPOT/NODEPOT        | Number of absorption transit compartments and whether to keep      |
|               |                                | the depot compartment                                              |
+---------------+--------------------------------+--------------------------------------------------------------------+
| LAGTIME       | :code:`OFF, ON`                | Absorption lagtime                                                 |
+---------------+--------------------------------+--------------------------------------------------------------------+
| COVARIATE     | `parameter`, `covariate`,      | Structural covariate effects (will always be added)                |
|               | `effect`                       |                                                                    |
+---------------+--------------------------------+--------------------------------------------------------------------+
| COVARIATE?    | `parameter`, `covariate`,      | Exploratory covariate effects (will be tested)                     |
|               | `effect`                       |                                                                    |
+---------------+--------------------------------+--------------------------------------------------------------------+
| DIRECTEFFECT  | :code:`LINEAR, EMAX, SIGMOID,  | Direct effect PD models.                                           |
|               | STEP, LOGLIN`                  |                                                                    |
+---------------+--------------------------------+--------------------------------------------------------------------+
| EFFECTCOMP    | :code:`LINEAR, EMAX, SIGMOID,  | Effect compartment PD models.                                      |
|               | STEP, LOGLIN`                  |                                                                    |
+---------------+--------------------------------+--------------------------------------------------------------------+
| INDIRECTEFFECT| :code:`LINEAR, EMAX, SIGMOID`  | Indirect effect PD models. `option` can be                         |
|               | , `option`                     | either :code:`PRODUCTION` or :code:`DEGRADATION`.                  |
+---------------+--------------------------------+--------------------------------------------------------------------+


.. _mfl_symbols:

Symbols
=======

The MFL supports certain automatically defined symbols that help with
automating feature declaration. Currently, the only use of such symbols lies in
declaring covariate effects via the syntax `COVARIATE(@SYMBOL, ...)`.

+-----------------+-------------+------------------------------------------------+
| Symbol          | Type        | Description of automatic definition            |
+=================+=============+================================================+
| `@IIV`          | Parameter   | All parameters with a corresponding IIV ETA    |
+-----------------+-------------+------------------------------------------------+
| `@PK`           | Parameter   | All PK parameters                              |
+-----------------+-------------+------------------------------------------------+
| `@PD`           | Parameter   | All PD parameters                              |
+-----------------+-------------+------------------------------------------------+
| `@PK_IIV`       | Parameter   | All PK parameters with a corresponding IIV ETA |
+-----------------+-------------+------------------------------------------------+
| `@PD_IIV`       | Parameter   | All PD parameters with a corresponding IIV ETA |
+-----------------+-------------+------------------------------------------------+
| `@ABSORPTION`   | Parameter   | All PK absorption parameters                   |
+-----------------+-------------+------------------------------------------------+
| `@ELIMINATION`  | Parameter   | All PK elimination parameters                  |
+-----------------+-------------+------------------------------------------------+
| `@DISTRIBUTION` | Parameter   | All PK distribution parameters                 |
+-----------------+-------------+------------------------------------------------+
| `@BIOAVAIL`     | Parameter   | All bioavailability parameters                 |
+-----------------+-------------+------------------------------------------------+
| `@CONTINUOUS`   | Covariate   | All continuous covariates                      |
+-----------------+-------------+------------------------------------------------+
| `@CATEGORICAL`  | Covariate   | All categorical covariates                     |
+-----------------+-------------+------------------------------------------------+


New symbols can be manually defined via the syntax `LET(SYMBOL, [...])`. The
same syntax can be used to override the automatic definition of a symbol. For
instance, to declare an explicit list of distribution parameters use
`LET(DISTRIBUTION, [V, K12, K21])`.


Describe intervals
==================

It is possible to use ranges and arrays to describe the search space for e.g. transit and peripheral compartments.

To test 0, 1, 2 and 3 peripheral compartments:

.. code::

    TRANSITS(0)
    TRANSITS(1)
    TRANSITS(2)
    TRANSITS(3)

This is equivalent to:

.. code::

    TRANSITS(0..3)

As well as:

.. code::

    TRANSITS([0, 1, 2, 3])

Redundant descriptions
======================

It is allowed to describe the same feature multiple times, however, this will not make any difference for which
features are described.

.. code::

    ABSORPTION(FO)
    ABSORPTION([FO, ZO])

This is equivalent to:

.. code::

    ABSORPTION([FO, ZO])

And:

.. code::

    PERIPHERALS(0..2)
    PERIPHERALS(1)

Is equivalent to:

.. code::

    PERIPHERALS(0..2)

Examples
========

An example of a search space for PK models with oral data:

.. code::

    ABSORPTION([ZO,SEQ-ZO-FO])
    ELIMINATION([MM,MIX-FO-MM])
    LAGTIME(ON)
    TRANSITS([0, 1, 3, 10],*)
    PERIPHERALS(0..1)

An example of a search space for PK models with IV data:

.. code::

    ELIMINATION([FO,MM,MIX-FO-MM])
    PERIPHERALS([0..2])


Search through all available absorption rates:

.. code::

    ABSORPTION(*)

Allow all combinations of absorption and elimination rates:

.. code::

    ABSORPTION(*)
    ELIMINATION(*)

All covariate effects on parameters with IIV:

.. code::

    COVARIATE(@IIV, @CONTINUOUS, *)
    COVARIATE(@IIV, @CATEGORICAL, CAT)

With fixed lists of covariates for which to add effects:

.. code::

    LET(CONTINUOUS, [AGE, WGT])
    LET(CATEGORICAL, SEX)
    COVARIATE(@IIV, @CONTINUOUS, *)
    COVARIATE(@IIV, @CATEGORICAL, CAT)

All continuous covariate effects of WGT on distribution parameters:

.. code::

   COVARIATE(@DISTRIBUTION, WGT, *)

Example for a PD model search space:

.. code::

    DIRECTEFFECT([linear, emax])
    EFFECTCOMP(*)
