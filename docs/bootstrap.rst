=========
Bootstrap
=========

Perform a bootstrap

~~~~~~~
Running
~~~~~~~

The bootstrap tool is available both in Pharmpy/pharmr and from the command line.

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools import read_modelfit_results, run_bootstrap

    model = read_model('path/to/model')
    model_results = read_modelfit_results('path/to/model')
    res = run_bootstrap(model=model, results=model_results, samples=300)

Example of running VPC from the command line:

.. code::

    pharmpy run bootstrap run1.mod --samples=300


~~~~~~~~~
Arguments
~~~~~~~~~

Mandatory
---------

+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| Argument                                          | Description                                                                             |
+===================================================+=========================================================================================+
| ``model``                                         | Start model                                                                             |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``results``                                       | ModelfitResults for the start model                                                     |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``samples``                                       | Number of resamples of the dataset. Minimum 1                                           |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+

Optional
--------

+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| Argument                                          | Description                                                                             |
+===================================================+=========================================================================================+
| ``dofv``                                          | Compute dOFV by evaluating the original dataset with the boostrap parameter estimates.  |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``strictness``                                    | Expression for which results to include in the calculations. Default include all.       |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+


.. math::

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Bootstrap postprocessing and results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameter statistics
--------------------

The ``parameter_statistics`` table contains summary statistics for over the bootstrap runs for the model parameters.

==================  =============================================
Column              Description
==================  =============================================
``mean``            Mean over all bootstrap runs
``median``          Median over all bootstrap runs
``bias``            Difference between the mean and
                    the value in the original model
``stderr``          Standard deviation over all bootstrap runs
``RSE``             Standard error divided by the mean
==================  =============================================


.. pharmpy-execute::
    :hide-code:

    from pharmpy.workflows.results import read_results
    res = read_results('tests/testdata/results/bootstrap_results.json')
    res.parameter_statistics


The ``parameter_distribution`` table gives a numeric overview of the distributions of the bootstrap parameter estimates.
For each parameter it contains the lowest and highest values, the median and values at some other selected percentiles.
All percentiles are calculated using linear interpolation if it falls between two data points. If the two data points are :math:`x_0`
and :math:`x_1` the percentile would be :math:`x_0 + (x_1 - x_0) f` where :math:`f` is :math:`[np]`, the fractional part of the number of observations
:math:`n` multiplied by the percentile :math:`p`.

.. pharmpy-execute::
    :hide-code:

    res.parameter_distribution

The ``parameter_estimates_histogram`` give histograms for the distributions of the parameter estimates:

.. pharmpy-execute::
    :hide-code:

    res.parameter_estimates_histogram


The raw parameter data is available in ``parameter_estimates``

.. pharmpy-execute::
    :hide-code:

    res.parameter_estimates


OFV statistics
--------------

Summary statistics for the objective function values of the bootstrap runs can be found in the ``ofv_statistics`` table, which has the following rows:

===========================  =============================================
Row                          Description
===========================  =============================================
``bootstrap_bootdata_ofv``   OFVs from the bootstrap runs
``original_bootdata_ofv``    Sum of iOFVs from original modelfit of individuals included in each bootstrap run
``bootstrap_origdata_ofv``   OFVs from all `dofv` runs, i.e. evaluations on original data on boostrap models  
``original_origdata_ofv``    OFV of original model
``delta_bootdata``           Difference between `original_bootdata_ofv` and `bootstrap_bootdata_ofv` for each model   
``delta_origdata``           Difference between `bootstrap_origdata_ofv` and the OFV of the original model 
===========================  =============================================

Note that some of these rows will not be created if the bootstrap was run without the `dofv` option.

.. pharmpy-execute::
    :hide-code:

    res.ofv_statistics

The ``ofv_distribution`` gives a numeric overview of the OFVs similar to the ``parameter_distriution`` described above. 

.. pharmpy-execute::
    :hide-code:

    res.ofv_distribution

A histogram of the bootstrap ofv from ``ofv_plot``:

.. pharmpy-execute::
    :hide-code:

    res.ofv_plot

The ``dofv_quantiles_plot`` show distribution of the delta-OFV metrics over the distribution quantiles. They are compared with
a chi-square distribution.

.. pharmpy-execute::
    :hide-code:

    res.dofv_quantiles_plot



The raw ofv data is available in ``ofvs``.

.. pharmpy-execute::
    :hide-code:

    res.ofvs


Covariance matrix
-----------------

A covariance matrix for the parameters is available in ``covariance_matrix``:

.. pharmpy-execute::
    :hide-code:

    res.covariance_matrix

Included individuals
--------------------

The ``included_individuals`` is a list of lists with all individuals that were included in each bootstrap run.

.. pharmpy-execute::
    :hide-code:

    import pandas as pd
    pd.DataFrame(res.included_individuals)
