==
QA
==

Pharmpy currently creates results after a PsN qa run.

~~~~~~~~~~~~~~
The qa results
~~~~~~~~~~~~~~

Overview
~~~~~~~~

The overview ``dofv`` table lists the most impactful modifications identified by QA.
More details for each result can be found in the individual sections.

The table shows an overview of possible modifications to different model aspects,
expected improements in OFV as well as number of additional parameters used.

.. pharmpy-execute::
    :hide-code:

    import pathlib
    from pharmpy.tools import read_results
    res = read_results(pathlib.Path('tests/testdata/results/qa_results.json'))
    res.dofv

Structural bias
~~~~~~~~~~~~~~~

This section aims at diagnosing the structural component of the model. It does so by estimating
the mean difference between model predictions and observations as a function of several independent variables.

The ``structural_bias`` table shows the ``CWRES`` and ``CPRED`` given different bins of the independent variable. 

.. pharmpy-execute::
    :hide-code:

    res.structural_bias

Fullblock
~~~~~~~~~

This section shows the effect of including a full block correlation struture in the base mdel.

The ``fullblock_parameters`` contains the estimated standard deviation (sd), correlation (corr) and
expected improvement in OFV after inclusion of a full block correlation structure of the random effects.

.. pharmpy-execute::
    :hide-code:

    res.fullblock_parameters


Boxcox
~~~~~~

This section shows the effect of applying a Box-Cox transformation to the ETA variables in the base model.

The ``boxcox_parameters`` contains the estimated shape parameter (Lambda) and expected improvment in OFV for a
Box-Cox transformation of the random effects.

.. pharmpy-execute::
    :hide-code:

    res.boxcox_parameters

Tdist
~~~~~

This section shows the effect of applying a t-distribution transformation to the ETA variables in the base model.

The ``tdist_parameters`` contains the estimated degrees of freedom and expected improvement in OFV for a 
t-distribution transformation of the random effects.

.. pharmpy-execute::
    :hide-code:

    res.tdist_parameters

Residual error
~~~~~~~~~~~~~~

This section shows the effect of including extended residual error models in the base model.

The ``residual_error`` table contains the residual error models, resulting expected improvement in OFV, 
required additional model parameters as well as their estimates.

.. pharmpy-execute::
    :hide-code:

    res.residual_error

Covariate effects
~~~~~~~~~~~~~~~~~

This section evaluates the impact of supplied covariates.

The ``covariate_effects`` table shows the expected improvement when including covariates.

.. pharmpy-execute::
    :hide-code:

    res.covariate_effects
