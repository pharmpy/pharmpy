|badge1| |badge2| |badge3| |badge4| |badge5| |badge6|

.. |badge1| image:: https://img.shields.io/pypi/v/pharmpy-core.svg
   :target: https://pypi.org/project/pharmpy-core

.. |badge2| image:: https://img.shields.io/pypi/l/pharmpy-core.svg
   :target: https://github.com/pharmpy/pharmpy/blob/main/LICENSE.LESSER

.. |badge3| image:: https://github.com/pharmpy/pharmpy/actions/workflows/main.yml/badge.svg
    :target: https://github.com/pharmpy/pharmpy/actions

.. |badge4| image:: https://img.shields.io/pypi/pyversions/pharmpy-core
   :target: https://www.python.org/downloads/

.. |badge5| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. |badge6| image:: https://codecov.io/gh/pharmpy/pharmpy/branch/main/graph/badge.svg?token=JZTHXXQPII
    :target: https://codecov.io/gh/pharmpy/pharmpy

.. _README:

|logo|
======

.. |logo| image:: https://github.com/pharmpy/pharmpy/raw/main/docs/Pharmpy_logo.svg
   :width: 250

Pharmpy is an open-source software package for pharmacometric modeling. It has functionality ranging from reading and
manipulating model files and datasets to full tools where subsequent results are collected and presented.

Features include:

* A **model abstraction** which splits a model into core components which Pharmpy understands and can manipulate:
  parameters, random variables, statements (including ODE system), dataset, and execution steps
* An **abstraction for modelfit results** which splits a parsed results into core components: e.g. OFV, parameter
  estimates, relative standard errors (RSEs), residuals, predictions
* **Functions for manipulation of models and datasets** in the modeling-module: e.g. change structural model, add
  time-after-dose column, add covariate effects
* **Tools for automated model development**: building various aspects (structural, iiv, iov, ruv, covariates, ...) of PK, PKPD, TMDD and drug-metabolite models automatically
* **Tools to aid model development** in the tools-module: execution of models within Python/R scripts, bootstrap, 
  comparison of estimation methods
* **Simplify scripting** of workflows. Makes it possible to run scripts including calls to long running tools multiple times without having to rerun already finished tool runs.
* Support for **multiple estimation tools**: parse NONMEM models, execute NONMEM, nlmixr2, and rxODE2 models, run all
  Pharmpy tools with NONMEM and some with nlmixr2

For more comprehensive information and documentation, see: https://pharmpy.github.io

Pharmpy can be used as a regular Python package, in R via the `pharmr <https://github.com/pharmpy/pharmr>`_ package,
or via its built in command line interface.

Getting started
===============

Installation
------------

For installation in R, see `pharmr <https://github.com/pharmpy/pharmr>`_.

Install the latest stable version from PyPI:

    pip install pharmpy-core    # or 'pip3 install' if that is your default python3 pip

Python Example
--------------

.. code-block:: none

   >>> from pharmpy.modeling import read_model
   >>> from pharmpy.tools import load_example_modelfit_results
   >>> model = load_example_model("pheno")
   >>> model.parameters
               value  lower upper    fix
   POP_CL   0.004693   0.00     ∞  False
   POP_VC   1.009160   0.00     ∞  False
   COVAPGR  0.100000  -0.99     ∞  False
   IIV_CL   0.030963   0.00     ∞  False
   IIV_VC   0.031128   0.00     ∞  False
   SIGMA    0.013086   0.00     ∞  False
   >>> res = load_example_modelfit_results("pheno")
   >>> res.parameter_estimates
   POP_CL     0.004696
   POP_VC     0.984258
   COVAPGR    0.158920
   IIV_CL     0.029351
   IIV_VC     0.027906
   SIGMA      0.013241
   Name: estimates, dtype: float64
   >>>

CLI Example
-----------

.. code-block:: none

    # Get help
    pharmpy -h

    # Remove first ID from dataset and save new model using new dataset
    pharmpy data filter run1.mod 'ID!=1'

    # Run tool for selecting IIV structure
    pharmpy run iivsearch run1.mod

User guide
----------

There is also a `user guide for getting started <https://pharmpy.github.io/latest/getting_started.html>`_

Contact
=======

This is the `team behind Pharmpy <https://pharmpy.github.io/latest/contributors.html>`_

Please ask a question in an issue or contact one of the maintainers if you have any questions.

Contributing
------------

If you interested in contributing to Pharmpy, you can find more information under
`Contribute <https://pharmpy.github.io/latest/contribute.html#contribute>`_.
