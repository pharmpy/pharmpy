.. image:: https://img.shields.io/pypi/v/pharmpy-core.svg
   :target: https://pypi.org/project/pharmpy-core

.. image:: https://img.shields.io/pypi/l/pharmpy-core.svg
   :target: https://github.com/pharmpy/pharmpy/blob/master/LICENSE.LESSER

.. image:: https://github.com/pharmpy/pharmpy/workflows/CI/badge.svg
    :target: https://github.com/pharmpy/pharmpy/actions

.. image:: https://img.shields.io/badge/python-3.7+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. image:: https://pepy.tech/badge/pharmpy-core/month
   :target: https://pepy.tech/project/pharmpy-core

.. _README:

.. highlight:: console

==============
Pharmpy |logo|
==============

.. |logo| image:: docs/Pharmpy_logo.svg
   :width: 100

https://pharmpy.github.io

.. start-longdesc

Pharmpy is a library for pharmacometrics. It can be used as a regular python package, in R via reticulate or via its built in command line interface.

Pharmpy is architectured to be able to handle different types of model formats and data formats and exposes a model agnostic API.

Current features:

* Parsing of many parts of a NONMEM model file
* Parsing of NONMEM result files
* CLI supporting dataset filtering, resampling, anonymization and viewing

Pharmpy is developed by the Uppsala University Pharmacometrics group.

.. end-longdesc

Installation
============

Install the latest stable version from PyPI::

    pip install pharmpy-core    # or 'pip3 install' if that is your default python3 pip

Python Example
==============


>>> from pharmpy import Model
>>> model = Model("run1.mod")
>>> print(model.modelfit_results.parameter_estimates)
THETA(1)      0.004696
THETA(2)      0.984258
THETA(3)      0.158920
OMEGA(1,1)    0.029351
OMEGA(2,2)    0.027906
SIGMA(1,1)    0.013241
Name: 2, dtype: float64
>>> model.parameters
       name     value  lower    upper    fix
   THETA(1)  0.004693   0.00  1000000  False
   THETA(2)  1.009160   0.00  1000000  False
   THETA(3)  0.100000  -0.99  1000000  False
 OMEGA(1,1)  0.030963   0.00       oo  False
 OMEGA(2,2)  0.031128   0.00       oo  False
 SIGMA(1,1)  0.013086   0.00       oo  False
>>>

R Example
=========

.. code-block:: r
   :skipif: False

>>> library(reticulate)
>>> use_python("python3")
>>> pharmpy <- import("pharmpy")
>>> model <- pharmpy$Model("run1.mod")
>>> model$modelfit_results$parameter_estimates
  THETA(1)   THETA(2)   THETA(3) OMEGA(1,1) OMEGA(2,2) SIGMA(1,1) 
0.00469555 0.98425800 0.15892000 0.02935080 0.02790600 0.01324100 
>>> model$parameters
       name     value  lower    upper    fix
   THETA(1)  0.004693   0.00  1000000  False
   THETA(2)  1.009160   0.00  1000000  False
   THETA(3)  0.100000  -0.99  1000000  False
 OMEGA(1,1)  0.030963   0.00       oo  False
 OMEGA(2,2)  0.031128   0.00       oo  False
 SIGMA(1,1)  0.013086   0.00       oo  False
>>>

CLI Example
===========

.. code-block:: none

    # Get help
    pharmpy -h

    # Remove first ID from dataset and save new model using new dataset
    pharmpy data filter run1.mod 'ID!=1'

    # Extract and print ofvs from multiple model runs
    pharmpy results ofv run*.mod


Development
===========

For information for contributors and developers see https://pharmpy.github.io/latest/development.html 
