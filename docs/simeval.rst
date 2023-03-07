=======
Simeval
=======

Pharmpy currently creates results after a PsN simeval run.

~~~~~~~~~~~~~~~~~~~
The simeval results
~~~~~~~~~~~~~~~~~~~

Sampled Individual OFVs
~~~~~~~~~~~~~~~~~~~~~~~

The ``sampled_iofv`` table contains the evaluated individual OFVs for each sampled dataset.

.. pharmpy-execute::
    :hide-code:

    import pathlib
    from pharmpy.tools.simeval.results import psn_simeval_results
    res = psn_simeval_results(pathlib.Path('tests/testdata/psn/simeval_dir1'))
    res.sampled_iofv

.. _individual ofv summary:

Individual OFV summary
~~~~~~~~~~~~~~~~~~~~~~~~

The ``iofv_summary`` table contain the iOFVs from the orginal model, the mean and standard deviation of the sampled iOFVs, the residual and a
residual outlier flag. The residual for each sample and ID is the distance from the observed iOFV to the simulated iOFV expressed in standard deviations of the simulated values.

.. math::

    \mathrm{res} = \frac{\mathrm{obs} - \operatorname{mean}(\mathrm{sim})}{\operatorname{sd}(\mathrm{sim})}

An individual is defined as an outlier if the corresponding residual is 3 or higher.

.. pharmpy-execute::
    :hide-code:

    res.iofv_summary

Individual prediction plot
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``individual_predictions_plot`` show PRED, IPRED and DV vs TIME (if available) for outlying individuals.

.. pharmpy-execute::
    :hide-code:

    res.individual_predictions_plot
