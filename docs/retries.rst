.. _retries:

=======
Retries
=======

The retries tool is a tool used to tweak the initial estimates of a given model. The initial estimates of all parameters
are changed slightly which can give better models or show instabilities for the model. When comparing and ranking the new
candidates, the best model is based solely on the 'OFV' value.


~~~~~~~
Running
~~~~~~~

The retries tool is available both in Pharmpy/pharmr.

An example of how to initiate the tool can be seen below

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools read_modelfit_results, run_structsearch

    start_model = read_model('path/to/model')
    start_model_results = read_modelfit_results('path/to/model')
    res = run_structsearch(model = start_model,
                            results = start_model_results,
                            number_of_candidate = 5,
                            scale = "normal")

Arguments
~~~~~~~~~
The arguments of the retries tool are listed below.

+-------------------------------------------------+---------------------------------------------------------------------+
| Argument                                        | Description                                                         |
+=================================================+=====================================================================+
| ``model``                                       | Start model to run retries on                                       |
+-------------------------------------------------+---------------------------------------------------------------------+
| ``results``                                     | ModelfitResults of the start model                                  |
+-------------------------------------------------+---------------------------------------------------------------------+
| ``strictness``                                  | :ref:`Strictness<strictness>` criteria for model selection.         |
|                                                 | Default is "minimization_successful or                              |
|                                                 | (rounding_errors and sigdigs>= 0.1)"                                |
+-------------------------------------------------+---------------------------------------------------------------------+
| ``number_of_candidates``                        | Number of retry-models to run. 5 is used as default                 |
+-------------------------------------------------+---------------------------------------------------------------------+
| ``fraction``                                    | Determine the allowed increase/decrease for the randomly generated  |
|                                                 | new initial estimates. Default is 0.1 (10%).                        |
+-------------------------------------------------+---------------------------------------------------------------------+
| ``use_initial_estimates``                       | Use initial parameter estimates when creating candidate models      |
|                                                 | instead of the final parameter estimates of the input model.        |
+-------------------------------------------------+---------------------------------------------------------------------+
| ``scale``                                       | :ref:`Scale<scales_retries>` to use when randomizing the initial    |
|                                                 | estimates. Currently supported scales are ``normal`` and ``UCP``.   |
+-------------------------------------------------+---------------------------------------------------------------------+
| ``prefix_name``                                 | String determining prefix of model names such that models are named |
|                                                 | {prefix_name}_retries_run2 for instance.                            |
+-------------------------------------------------+---------------------------------------------------------------------+
| ``seed``                                        | A random number generator or seed to use for random sampling.       |
+-------------------------------------------------+---------------------------------------------------------------------+

.. _scales_retries:

~~~~~~~~~~
Scales
~~~~~~~~~~

There are different ways for the tool of computing the retry models. A shared behaviour however, is that all new initial estimates
of parameters are randomly generated. The available scales can be seen in the table below.

+---------------------------+----------------------------------------------------------------------------------------+
| scales                    | Description                                                                            |
+===========================+========================================================================================+
| ``'normal'``              | Will iteratively try out new, randomly generated initial estimates. If the covariance  |
|                           | matrix for OMEGA(s) and SIGMA(s) is determined to not be positive semi-definite, new   |
|                           | random initial estimates is generated (maximum of 20 times)                            |
+---------------------------+----------------------------------------------------------------------------------------+
| ``'UCP'``                 | All parameters get new are transformed to the Unconstrained Parameter Scale (UCP scale)|
|                           | where new initial values are generated before inverting the transformation. Through    |
|                           | this method, the values are only required to be randomized once.                       |
+---------------------------+----------------------------------------------------------------------------------------+

Normal
~~~~~~

Using the normal scale, the tool is similar to the one found in PsN's tool parallelle retries. For all parameters, the given initial
estimates is used as a center point for a uniform distribution. The new random initial value is then taken from a uniform 
distribution with bound 10% above and below the given initial estimate (or the given upper or lower bound). The new value
is allowed to be at most 10^(⁻6) units from any of the bounds. This is performed for all parameters (THETAs, OMEGAs and 
SIGMAs).

Once all new initial values have been computed, the covariance matrix (made from OMEGAs and SIGMAs) will be checked to see
if it's positive semidefinite or not. If it is, the new retry model will be fitted. If not, new initial values will be 
generated for all parameters as described above. This is performed a maxiumum of 20 times. If a positive semi definite 
covariance matrix has not been found by then, an error is raised.

UCP
~~~~~~~~~~~~~

As opposed to the normal scale, the UCP scale will always return a model with functioning initial
values for all parameters. This is ensured through the use of Unconstrained Parameters. All parameters are transformed
to Unconstrained Parameters. In this value space, new initial values are computed with the same allowed variance, allowing
values to be taken from a uniform distribution of values from 10% below, to 10% above the current initial estimate. With 
these changes applied, the inverse Unconstrained Parameters transformation is performed.

As the Unconstrained Parameter Scale is different for each parameter, the relative change might be different 
for all parameters and is not limited to +/- 10% of the given inital estimate.   

~~~~~~~~~~~~~~~~~~~
The Retries results
~~~~~~~~~~~~~~~~~~~

The results object contains various summary tables which can be accessed in the results object, as well as files in
.csv/.json format. The name of the selected best model (based on the input selection criteria) is also included.

An example of an entire retries run can be seen below

.. pharmpy-code::

    start_model = read_model('path/to/model')
    start_model_results = read_modelfit_results('path/to/model')
    res = run_structsearch(model = start_model,
                            results = start_model_results,
                            number_of_candidate = 5,
                            fraction = 0.1,
                            scale = "UCP")

The ``summary_tool`` table contains information of the model results and final ranking. It also contains information
regarding how many attempts it took for the model to successfully find a positive semi definite covariance matrix:

.. pharmpy-execute::
   :hide-code:

    from pharmpy.workflows.results import read_results
    res = read_results('tests/testdata/results/retries_results.json')
    res.summary_tool

To see information about the actual model runs, such as minimization status, estimation time, and parameter estimates,
you can look at the ``summary_models`` table. The table is generated with
:py:func:`pharmpy.tools.summarize_modelfit_results`.

.. pharmpy-execute::
    :hide-code:

    res.summary_models

Finally, you can see a summary of different errors and warnings in ``summary_errors``.
See :py:func:`pharmpy.tools.summarize_errors` for information on the content of this table.

.. pharmpy-execute::
    :hide-code:

    import pandas as pd
    pd.set_option('display.max_colwidth', None)
    res.summary_errors
