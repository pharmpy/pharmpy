========
Crossval
========

Pharmpy can currently create results after a PsN crossval run.

~~~~~~~~~~~~~~~~~~~~
The crossval results
~~~~~~~~~~~~~~~~~~~~

Crossvalidation OFVs
~~~~~~~~~~~~~~~~~~~~

The ``runs`` table contains the OFVs of the estimation and prediction runs of crossval.

.. pharmpy-execute::
    :hide-code:

    import pathlib
    from pharmpy.tools.crossval.results import psn_crossval_results
    res = psn_crossval_results(pathlib.Path('tests/testdata/psn/crossval_dir1'))
    res.runs

The sum of all prediction OFVs can be found in ``prediction_ofv_sum``

.. pharmpy-execute::
    :hide-code:

    res.prediction_ofv_sum
