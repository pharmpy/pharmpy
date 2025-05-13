.. figure:: Pharmpy_logo.svg
   :class: only-light no-scaled-link
   :width: 400
   :align: center

.. figure:: Pharmpy_logo_dark.svg
   :class: only-dark no-scaled-link
   :width: 400
   :align: center

==================
Welcome to Pharmpy
==================


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

Pharmpy can be used as a regular Python package, in R via the pharmr package, or via its built in command line
interface.

We encourage your contribution to the Pharmpy project! You can report issues and post suggestions of features via
GitHub issues (to `Pharmpy <https://github.com/pharmpy/pharmpy/issues>`_ or to
`pharmr <https://github.com/pharmpy/pharmr/issues>`_). If you want to contribute with code you can find more information
under :ref:`contribute`.

Pharmpy is maintained by the Uppsala University Pharmacometrics group and is an open-source project.

.. toctree::
   :hidden:

   About us <contributors>
   getting_started
   user_guide
   api
   news
   contribute
   References and citation <citation>
   license
   developers
