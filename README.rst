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

Simplify and Accelerate Your Pharmacometric Workflows with Pharmpy
==================================================================

Developed by `Uppsala University Pharmacometrics Group <https://www.farmaci.uu.se/research/pharmacometrics/>`_, Pharmpy is an open-source toolkit designed for pharmacometrics. It supports Python, R via `pharmr <https://github.com/pharmpy/pharmr>`_, and CLI. Get to know the `Pharmpy Team <https://pharmpy.github.io/latest/contributors.html>`_!

Switch Effortlessly Between Software with Pharmpy's Software-Agnostic Model Abstraction
---------------------------------------------------------------------------------------

* **Model Abstraction**: Pharmpy acts as a middle layer, enabling seamless transitions between different software for pharmacometrics.
* **Model Manipulation**: Make changes to your model's components without worrying about software compatibility.
* **Result Extraction**: Retrieve your model's estimation results, regardless of the original software used.

Current support for NONMEM, nlmixr2, and rxODE2
-----------------------------------------------

* **NONMEM Support**: Read and interpret NONMEM models seamlessly.
* **Workflow Automation**: Automate complex workflows in NONMEM and partially in nlmixr2.
* **Plugin System**: Extend support for model formats like NONMEM, nlmixr2, and rxODE2.
* **Model Conversion**: Convert models between supported formats effortlessly.

.. end-longdesc

Getting started
===============

Pharmpy can be used in a Python program as a library or via its command line interface.
It can also be used via its R wrapper package: `pharmr <https://github.com/pharmpy/pharmr>`_.

Installation
------------

Install the latest stable version from PyPI::

    pip install pharmpy-core    # or 'pip3 install' if that is your default python3 pip

To be able to use components using machine learning the tflite package is needed. It can
be installed using::

    pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtime

Python Example
--------------

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
-----------

.. code-block:: none

    # Get help
    pharmpy -h

    # Remove first ID from dataset and save new model using new dataset
    pharmpy data filter run1.mod 'ID!=1'

    # Extract and print ofvs from multiple model runs
    pharmpy results ofv run*.mod

User Guide
==========

In the `User Guide <https://pharmpy.github.io/latest/user_guide.html>`_ you can find information on how to use Pharmpy in Python and R as well as from the command line.
This guide covers basic use cases in Pharmpy.

Contact Information
===================

We love hearing from the community! For general questions, feel free to reach out to one of our maintainers:

- **Rikard Nordgren**: `rikard.nordgren@farmaci.uu.se <mailto:rikard.nordgren@farmaci.uu.se>`_ (Maintainer)
- **Stella Belin**: `stella.belin@farmaci.uu.se <mailto:stella.belin@farmaci.uu.se>`_ (Maintainer)

Support or Technical Questions
------------------------------

You can report issues and post suggestions of features via GitHub issues (to `Pharmpy <https://github.com/pharmpy/pharmpy/issues>`_ or to `pharmr <https://github.com/pharmpy/pharmr/issues>`_).

Contributing
------------

If you want to contribute with code, you can find more information under `Contribute <https://pharmpy.github.io/latest/contribute.html#contribute>`_.
