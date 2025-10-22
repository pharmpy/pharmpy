==
QA
==

Pharmpy currently both has its own implementation of parts of the QA tool and can read the results from a PsN run.

~~~~~~~
Running
~~~~~~~

The QA tool is available both in Pharmpy/pharmr and from the command line.


.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools import read_modelfit_results, run_qa

    model = read_model('path/to/model')
    model_results = read_modelfit_results('path/to/model')
    res = run_qa(model=model, results=model_results, linearize=False)


Example of running QA from the command line:

.. code::

    pharmpy run qa run1.mod


~~~~~~~~~
Arguments
~~~~~~~~~

+------------------+------------------------------------------------------------------------+
| Argument         | Description                                                            |
+==================+========================================================================+
| ``model``        | Input model                                                            |
+------------------+------------------------------------------------------------------------+
| ``results``      | ModelfitResults for the input model                                    |
+------------------+------------------------------------------------------------------------+
| ``linearize``    | Whether to linearize input model before analysis                       |
+------------------+------------------------------------------------------------------------+
| ``skip``         | Sections to skip (see :ref:`Sections<qa_sections>` for valid options)  |
+------------------+------------------------------------------------------------------------+

.. _qa_sections:

~~~~~~~~
Sections
~~~~~~~~

Currently, Pharmpy supports parts of the PsN QA tool, and more will be added. See `PsN`_ documentation for full
description of the QA tool sections.

.. _PsN: https://uupharmacometrics.github.io/PsN/docs.html

The following sections are currently supported:

* Full OMEGA block (:code:`'fullblock'`)
* Box-Cox transformation (:code:`'boxcox'`)
* t-distribution (:code:`'tdist'`)


~~~~~~~~~~~~~~
The QA results
~~~~~~~~~~~~~~

.. note::

    Since Pharmpy both has its own QA implementation and can read in PsN results, some
    sections are only applicable to PsN.

Overview
~~~~~~~~

The overview ``dofv`` table lists the most impactful modifications identified by QA.
More details for each result can be found in the individual sections.

The table shows an overview of possible modifications to different model aspects,
expected improvements in OFV as well as number of additional parameters used.

.. pharmpy-execute::
    :hide-code:

    import pathlib
    from pharmpy.tools import read_results
    psn_res = read_results(pathlib.Path('tests/testdata/results/qa_psn_results.json'))
    res = read_results(pathlib.Path('tests/testdata/results/qa_results.json'))
    psn_res.dofv

Structural bias
~~~~~~~~~~~~~~~

.. note::

    Only available from PsN run.

This section aims at diagnosing the structural component of the model. It does so by estimating
the mean difference between model predictions and observations as a function of several independent variables.

The ``structural_bias`` table shows the ``CWRES`` and ``CPRED`` given different bins of the independent variable. 

.. pharmpy-execute::
    :hide-code:

    psn_res.structural_bias

Fullblock
~~~~~~~~~

This section shows the effect of including a full block correlation structure in the base model.

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

In Pharmpy, the following plots are generated showing the transformed and untransformed distribution for each
eta:

.. pharmpy-execute::
    :hide-code:

    res.boxcox_plot



Tdist
~~~~~

This section shows the effect of applying a t-distribution transformation to the ETA variables in the base model.

The ``tdist_parameters`` contains the estimated degrees of freedom and expected improvement in OFV for a 
t-distribution transformation of the random effects.

.. pharmpy-execute::
    :hide-code:

    psn_res.tdist_parameters

In Pharmpy, the following plots are generated showing the transformed and untransformed distribution for each
eta (sampling values from distribution with 1000 samples):

.. pharmpy-execute::
    :hide-code:

    res.tdist_plot


Residual error
~~~~~~~~~~~~~~

.. note::

    Only available from PsN run.

This section shows the effect of including extended residual error models in the base model.

The ``residual_error`` table contains the residual error models, resulting expected improvement in OFV, 
required additional model parameters as well as their estimates.

.. pharmpy-execute::
    :hide-code:

    psn_res.residual_error

Covariate effects
~~~~~~~~~~~~~~~~~

.. note::

    Only available from PsN run.

This section evaluates the impact of supplied covariates.

The ``covariate_effects`` table shows the expected improvement when including covariates.

.. pharmpy-execute::
    :hide-code:

    psn_res.covariate_effects
