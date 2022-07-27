.. image:: https://img.shields.io/pypi/v/pharmpy-core.svg
   :target: https://pypi.org/project/pharmpy-core

.. image:: https://img.shields.io/pypi/l/pharmpy-core.svg
   :target: https://github.com/pharmpy/pharmpy/blob/main/LICENSE.LESSER

.. image:: https://github.com/pharmpy/pharmpy/workflows/CI/badge.svg
    :target: https://github.com/pharmpy/pharmpy/actions

.. image:: https://img.shields.io/pypi/pyversions/pharmpy-core
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. image:: https://codecov.io/gh/pharmpy/pharmpy/branch/main/graph/badge.svg?token=JZTHXXQPII
    :target: https://codecov.io/gh/pharmpy/pharmpy

.. image:: https://pepy.tech/badge/pharmpy-core/month
   :target: https://pepy.tech/project/pharmpy-core

.. _README:

.. highlight:: console

======
|logo|
======

.. |logo| image:: docs/Pharmpy_logo.svg
   :width: 250

https://pharmpy.github.io

.. start-longdesc

Pharmpy is a library and toolkit for pharmacometrics. It can be used as a regular Python package, in R
via the `pharmr <https://github.com/pharmpy/pharmr>`_ package or via its built in command
line interface.

Current features:

* A model abstraction as a foundation for higher level operations on models
* Functions for manipulation of models, e.g. changing model components like elimination or absorption
* Reading NONMEM models and results
* Running models and complex workflows (with NONMEM or to some extent nlmixr)

This is the `team behind Pharmpy <https://pharmpy.github.io/latest/contributors.html>`_

.. end-longdesc

Installation
============

Install the latest stable version from PyPI::

    pip install pharmpy-core    # or 'pip3 install' if that is your default python3 pip

To be able to use components using machine learning the tflite package is needed. It can
be installed using::

    pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtime

Python Example
==============

>>> from pharmpy.modeling import read_model
>>> model = read_model("run1.mod")
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

CLI Example
===========

.. code-block:: none

    # Get help
    pharmpy -h

    # Remove first ID from dataset and save new model using new dataset
    pharmpy data filter run1.mod 'ID!=1'

    # Extract and print ofvs from multiple model runs
    pharmpy results ofv run*.mod
