=========================
Case deletion diagnostics
=========================

Pharmpy currently creates results after a PsN cdd run.

~~~~~~~~~~~~~~~
The cdd results
~~~~~~~~~~~~~~~

Case results
~~~~~~~~~~~~

The ``case_results`` table contains the different results and metrics for each case.


.. pharmpy-execute::
    :hide-code:

    from pharmpy.results import read_results
    res = read_results('tests/testdata/results/cdd_results.json')
    res.case_results

Cook score
----------

The Cook score for each case is calculated as:

.. math::

    \sqrt{(P_i - P_{orig})^T \operatorname{cov}(P_{orig})^{-1} (P_i - P_{orig})}

Where :math:`P_i` is the estimated parameter vector for case :math:`i`, :math:`P_{orig}` is the estimated parameter vector for the original model and :math:`\operatorname{cov}(P_{orig})` is the covariance matrix of the estimated parameters.

Jackknife cookscore
-------------------

This is the same as the Cook score above, but instead using the Jackknife covariance matrix.

.. math::

    \sqrt{(P_i - P_{orig})^T \operatorname{cov}^{\operatorname{jackknife}}(P_{orig})^{-1} (P_i - P_{orig})}

where

.. math::

    \operatorname{cov}_{j,k}^{\operatorname{jackknife}} = \frac{N - 1}{N}\sum_{i=1}^N(p_{i,j} - \overline{p}_j)(p_{i,k} - \overline{p}_k)

is the jackknife estimate of the covariance between :math:`p_{orig,j}` and :math:`p_{orig,k}` which is used to calculate the
full jackknife covariance matrix.

.. math::

    \overline{p}_j = \frac{1}{N}\sum_{i=1}^N p_{i,j}

is the mean of parameter :math:`p_{i,j}` over all case deleted datasets. :math:`j` being the index in the parameter vector and :math:`i` being the case index. 

Covariance ratio
----------------

The covariance ratio for each case is calculated as:

.. math::

    \sqrt{\frac{\operatorname{det}({\operatorname{cov}(P_i))}}{\operatorname{det}(\operatorname{cov}(P_{orig}))}}

Delta OFV
---------

For the delta OFV to be calculated the cases must correspond to individuals. Then it is calculated as

.. math::

    dOFV = OFV_{all} - iOFV_{k} - OFV_{k}

where :math:`OFV_{all}` is the OFV of the full run with all individuals included, :math:`iOFV_k`
is the individual OFV of the k:th individual in the full run and :math:`OFV_k` is the OFV of the run
with the k:th individual removed. [dOFV]_

Skipped individuals
-------------------

A list of the individuals that were skipped for each case.

Case column
~~~~~~~~~~~

The Name of the case column in the dataset can be found in ``case_column``.

.. pharmpy-execute::

    res.case_column

References
~~~~~~~~~~

.. [dOFV] Rikard Nordgren, Sebastian Ueckert, Svetlana Freiberga and Mats O. Karlsson, "Faster methods for case deletion diagnostics: dOFV and linearized dOFV", PAGE 27 (2018) Abstr 8683 https://www.page-meeting.org/?abstract=8683
