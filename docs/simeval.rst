=======
Simeval
=======

Pharmpy currently creates results after a PsN simeval run.

~~~~~~~~~~~~~~~~~~~
The simeval results
~~~~~~~~~~~~~~~~~~~

Individual OFVs
~~~~~~~~~~~~~~~

The `iofv` table contains the estimated individual OFV of the original model and the evaluated individual OFVs for each sampled dataset.

.. jupyter-execute::
    :hide-code:

    import pathlib
    from pharmpy.methods.simeval.results import psn_simeval_results
    res = psn_simeval_results(pathlib.Path('tests/testdata/psn/simeval_dir1'))
    res.iofv

Individual OFV residuals
~~~~~~~~~~~~~~~~~~~~~~~~

The residual for each sample and ID is the distance from the observed iOFV to the simulated iOFV expressed in standard deviations of the simulated values.

.. math::

    \mathrm{res} = \frac{\mathrm{obs} - \mathrm{sim}}{\mathop{sd}(\mathrm{sim})}

The residuals are stored in the `iofv_residuals` table.

.. jupyter-execute::
    :hide-code:

    res.iofv_residuals

Dataset flag
~~~~~~~~~~~~

In `data_flag` a flag for each dataset item is generated to indicate the outlier status. The meanings of the flag values are:

=====  ==============
Value  Meaning
=====  ==============
0      Not an outlier
1      iOFV outlier
2      CWRES outlier
=====  ==============

.. jupyter-execute::
    :hide-code:

    res.data_flag

Residual outliers
~~~~~~~~~~~~~~~~~

`residual_outliers` is a list of individuals that are outliers based on the residual iOFV.
