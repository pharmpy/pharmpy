.. _selection_criteria:

==================
Selection criteria
==================

Pharmpy tools support multiple different selection criteria, and most tools allow the user to specify which to use.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Objective function value (OFV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses the objective function value as a selection critera, most often
:math:`-2\text{log}\mathcal{L}(\hat{\theta} \mid y)`.

~~~~~~~~~~~~~~~~~~~~~~~~~~~
Likelihood ratio test (LRT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Performs a likelihood ratio test, where :math:`\alpha` is the cutoff p-value.

.. note::

    In tools that use likelihood ratio test (e.g. COVSearch and RUVSearch) the models are always nested, and thus
    LRT is applicable. However, if this is set for some of the other tools this might not be the case, but currently
    Pharmpy does not detect whether models are nested. Thus, if using this in another tool, LRT might not be applicable
    for all models in all tools.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Aikaike information criterion (AIC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Aikaike information criterion can also be calculated. In order to calculate the AIC in Pharmpy standalone, you can
use the :py:func:`pharmpy.modeling.calculate_aic`-function. The AIC is described as follows:

.. math::

    AIC = -2\text{log}\mathcal{L}(\hat{\theta} \mid y) + 2dim({\theta})


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Bayesian information criterion (BIC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pharmpy supports multiple types of BIC. In order to calculate the BIC in Pharmpy standalone, you can use the
:py:func:`pharmpy.modeling.calculate_bic`-function.

Fixed
-----

For linear models, the BIC :footcite:p:`schwarz_1978` is described as follows:

.. math::

    BIC_{fixed} = -2\text{log}\mathcal{L}(\hat{\theta} \mid y) + dim(\theta)\text{log} n_{tot}

Where :math:`\theta` is estimated parameters and :math:`n_{tot}` is the number of observations.

Mixed
-----

For mixed effect models, BIC takes the number of random vs. fixed parameters into consideration
:footcite:p:`delattre_2014`, and is described as follows:

.. math::

    BIC_{mixed} = -2\text{log}\mathcal{L}(\hat{\theta} \mid y) + dim({\theta_R})\text{log} N + dim({\theta_F})\text{log} n_{tot}

Where :math:`\theta_R` is random parameters, :math:`N` is the number of subjects, :math:`\theta_F` is fixed parameters,
and :math:`n_{tot}` is the number of observations.

IIV
---

The BIC for IIV :footcite:p:`delattre_2020` is described as follows:

.. math::

    BIC_{IIV} = -2\text{log}\mathcal{L}(\hat{\theta} \mid y) + dim({\omega_{IIV}})\text{log} N

Where :math:`\omega_{IIV}` is IIV parameters and :math:`N` is the number of subjects.

Random
------

The BIC for random effects is described as follows:

.. math::

    BIC_{random} = -2\text{log}\mathcal{L}(\hat{\theta} \mid y) + dim({\theta})\text{log} N

Where :math:`\theta` is parameters and :math:`N` is the number of subjects.


Modified BIC (mBIC)
-------------------

The mBIC selection criterion :footcite:p:`chen_2025` is an additional penalty term, and is combined with any of the above mentioned
types of BIC. mBIC was introduced to address the problem of multiple testing, and penalizes against the size of the
:ref:`search space<mfl>` and the complexity of the candidate model.

Currently, mBIC supports evaluation of the structural PK model and the IIV model, and there are two different penalty
calculations for each of them. This means that when e.g. evaluation the structural model, the ModelSearch tool only
uses mBIC for the structural model.

.. note::

    mBIC is currently only supported by ModelSearch, IIVSearch, and IOVSearch.

mBIC for the structural model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In mBIC for the structural model, different components of the model structure penalizes differently (see table below).
For the structural model, we can define :math:`mBIC_{structural}` with the following equation:

.. math::

    mBIC_{structural} = BIC_{mixed} + 2 k_M \text{log}\frac{M}{E}

Where :math:`k_M` is the number of structural predictors in the candidate model, :math:`M` is the number of structural
predictors in the search space, and :math:`E` is the `a priori` expected number of predictors. The :math:`E`-value is
set by the user, and is bounded by :math:`0 < E < M` (where 0 is no expected number of predictors and thus infinite
penalty, and :math:`M` is all predictors in the search space which means no penalty).

The number of predictors is defined accordingly:

+--------------------------------+---------------------------+------------------------------+
| Feature                        | Contribution to :math:`M` | Contribution to :math:`k_M`  |
+================================+===========================+==============================+
| :code:`ABSORPTION(FO)`         | 0                         | 0                            |
+--------------------------------+---------------------------+------------------------------+
| :code:`ABSORPTION(ZO)`         | 0                         | 0                            |
+--------------------------------+---------------------------+------------------------------+
| :code:`ABSORPTION(SEQ-ZO-FO)`  | 1                         | 1                            |
+--------------------------------+---------------------------+------------------------------+
| :code:`LAGTIME(OFF)`           | 0                         | 0                            |
+--------------------------------+---------------------------+------------------------------+
| :code:`LAGTIME(ON)`            | 1                         | 1                            |
+--------------------------------+---------------------------+------------------------------+
| :code:`TRANSITS(0..x)`         | :math:`x`                 | 1                            |
+--------------------------------+---------------------------+------------------------------+
| :code:`PERIPHERALS(0..x)`      | :math:`x`                 | :math:`x`                    |
+--------------------------------+---------------------------+------------------------------+
| :code:`ELIMINATION(FO)`        | 0                         | 0                            |
+--------------------------------+---------------------------+------------------------------+
| :code:`ELIMINATION(MM)`        | 0                         | 0                            |
+--------------------------------+---------------------------+------------------------------+
| :code:`ELIMINATION(MIX-FO-MM)` | 1                         | 0                            |
+--------------------------------+---------------------------+------------------------------+

We can view this as four different categories:

- Mutually exclusive: one of options have to be selected
    - FO/ZO absorption, FO/MM elimination
    - Contribution to :math:`M` and :math:`k_M` are each 0
- On/off: whether a component is active or not
    - Lagtime or no lag time
    - Contribution to :math:`M` and :math:`k_M` are each 0 when "off", 1 when "on"
- Stackable: combinations that can be stacked
    - Peripherals, going from FO/ZO absorption to SEQ-ZO-FO, going from FO/MM elimination to MIX-FO-MM
    - Contribution to :math:`M` and :math:`k_M` are both increased increased by the number of options (e.g. having
      0-2 peripherals in the search space will give :math:`M=2` and :math:`k_M` of how many peripherals is in candidate)
- Non-stackable: numeric option which is mutually exclusive
    - Transits
    - Contribution to :math:`M` is the number of non :math:`k_M` are both increased increased by the number of
      non-stackable options (e.g. having 0-2 transits in the search space will give :math:`M=2` and :math:`k_M` of 1
      in any candidate with transits)

As an example, consider the default search space for an oral PK drug in the AMD tool, and some example candidates:

.. code-block::

    ABSORPTION([FO,ZO,SEQ-ZO-FO])
    ELIMINATION(FO)
    LAGTIME([OFF,ON])
    TRANSITS([0,1,3,10],*)
    PERIPHERALS([0,1])

    M = 6

.. tabs::

    .. tab:: Candidate example 1

        .. code-block::

            ABSORPTION(FO)
            ELIMINATION(FO)
            LAGTIME(OFF)
            TRANSITS(0)
            PERIPHERALS(1)

            k_M = 1

    .. tab:: Candidate example 2

         .. code-block::

            ABSORPTION(SEQ-ZO-FO)
            ELIMINATION(FO)
            LAGTIME(OFF)
            TRANSITS(1,DEPOT)
            PERIPHERALS(1)

            k_M = 3

    .. tab:: Candidate example 3

         .. code-block::

            ABSORPTION(SEQ-ZO-FO)
            ELIMINATION(FO)
            LAGTIME(ON)
            TRANSITS(1,DEPOT)
            PERIPHERALS(2)

            k_M = 4

If the search space includes transits but without depot, the contribution to :math:`M` is still the same as if the
depot was kept:

.. code-block::

    ABSORPTION([FO,ZO,SEQ-ZO-FO])
    ELIMINATION(FO)
    LAGTIME([OFF,ON])
    TRANSITS([0,1,3,10],NODEPOT)
    PERIPHERALS([0,1])

    M = 6

If there is only one value in a category, the contribution to :math:`M` and :math:`k_M` will both be 0, no matter
the value (since no choice is being made):

.. code-block::

    ABSORPTION([FO,ZO,SEQ-ZO-FO])
    ELIMINATION(FO)
    LAGTIME([OFF,ON])
    TRANSITS([0,1,3,10],*)
    PERIPHERALS(1)

    M = 5

.. tabs::

    .. tab:: Candidate example 1

        .. code-block::

            ABSORPTION(FO)
            ELIMINATION(FO)
            LAGTIME(OFF)
            TRANSITS(0)
            PERIPHERALS(1)

            k_M = 0

    .. tab:: Candidate example 2

         .. code-block::

            ABSORPTION(SEQ-ZO-FO)
            ELIMINATION(FO)
            LAGTIME(ON)
            TRANSITS(0)
            PERIPHERALS(1)

            k_M = 2

    .. tab:: Candidate example 3

         .. code-block::

            ABSORPTION(SEQ-ZO-FO)
            ELIMINATION(FO)
            LAGTIME(ON)
            TRANSITS(0)
            PERIPHERALS(2)

            k_M = 5


mBIC for the IIV model
^^^^^^^^^^^^^^^^^^^^^^

In mBIC for the IIV model, the penalty consists of one part for the diagonal omegas and one part for the off-diagonals.
For the IIV model, we can define :math:`mBIC_{IIV}` with the following equation:

.. math::

    mBIC_{IIV} = BIC_{IIV} + 2 k_p \text{log}\frac{p}{E_p} + 2 k_q \text{log}\frac{q}{E_q}

Where :math:`k_p` and :math:`k_q` are the number of diagonal and off-diagonal omega predictors in the candidate model,
:math:`p` and :math:`q` are the number of diagonal and off-diagonal omega predictors in the search space, and
:math:`E_p` and :math:`E_q` is the `a priori` expected number of diagonal and off-diagonal omega predictors.

If you're only testing the number of diagonal omegas, the penalty related to the off-diagonal omegas is omitted.

~~~~~~~~~~~~~~~~~~~~~~
AMD selection criteria
~~~~~~~~~~~~~~~~~~~~~~

Different subtools in AMD uses different selection criteria, mostly depending on whether the candidate models are
nested or not.

+---------------+--------------------+
| Subtool       | Selection criteria |
+===============+====================+
| ModelSearch   | BIC (mixed)        |
+---------------+--------------------+
| StructSearch  | BIC (mixed)        |
+---------------+--------------------+
| IIVSearch     | BIC (IIV)          |
+---------------+--------------------+
| RUVSearch     | LRT                |
+---------------+--------------------+
| IOVSearch     | BIC (random)       |
+---------------+--------------------+
| COVSearch     | LRT                |
+---------------+--------------------+


~~~~~~~~~~
References
~~~~~~~~~~

.. footbibliography::
