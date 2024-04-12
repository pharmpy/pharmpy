|badge1| |badge2| |badge3| |badge4| |badge5| |badge6| |badge7|

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

.. |badge7| image:: https://img.shields.io/pypi/dm/pharmpy-core.svg
   :target: https://pypistats.org/packages/pharmpy-core

.. _README:

.. highlight:: console

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


.. end-longdesc

Getting started
===============

Installation
------------

For installation in R see `pharmr <https://github.com/pharmpy/pharmr>`_. 

Install the latest stable version from PyPI::

    pip install pharmpy-core    # or 'pip3 install' if that is your default python3 pip

To be able to use components using machine learning the tflite package is needed. It can
be installed using::

    pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtime

Python Example
--------------


.. code-block:: none

   >>> from pharmpy.modeling import read_model, read_modelfit_results
   >>> model = read_model("run1.mod")
   >>> res = read_modelfit_results("run1.mod")
   >>> res.parameter_estimates
   THETA_1       0.004696
   THETA_2       0.984258
   THETA_3       0.158920
   OMEGA_1_1     0.029351
   OMEGA_2_2     0.027906
   SIGMA_1_1     0.013241
   Name: ests, dtype: float64
   >>> model.parameters
            name     value   lower    upper    fix
        THETA_1     0.004693  0.00  1000000  False
        THETA_2     1.009160  0.00  1000000  False
        THETA_3     0.100000 -0.99  1000000  False
      OMEGA_1_1     0.030963  0.00       oo  False
      OMEGA_2_2     0.031128  0.00       oo  False
      SIGMA_1_1     0.013086  0.00       oo  False
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


Contact
=======

This is the `team behind Pharmpy <https://pharmpy.github.io/latest/contributors.html>`_

Please ask a question in an issue or contact one of the maintainers if you have any questions.

Contributing
------------

If you interested in contributing to Pharmpy, you can find more information under `Contribute <https://pharmpy.github.io/latest/contribute.html#contribute>`_.
