.. image:: https://img.shields.io/pypi/v/pharmpy-core.svg
   :target: https://pypi.org/project/pharmpy-core

.. image:: https://img.shields.io/pypi/l/pharmpy-core.svg
   :target: https://github.com/pharmpy/pharmpy/blob/master/LICENSE.LESSER

.. image:: https://github.com/pharmpy/pharmpy/workflows/CI/badge.svg
    :target: https://github.com/pharmpy/pharmpy/actions

.. image:: https://img.shields.io/badge/python-3.6+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

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

Testing
-------

To run the all tests via Pytest_ install Tox_::

    pip3 install tox

Then run::

    tox

Note, to combine the coverage data from all the Tox_ environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox

Missing a Python interpreter? Ubuntu 18.04 and no more ``python3.5``? No worries! Do this::

    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install python3.5-dev

Don't worry. Everything will be prefixed ``python3.5`` so there won't be any collision with e.g.
``python3`` (it'll still use the system standard).

Documentation
-------------

Local documentation found in ``dist/docs/``. To build, run::

    tox -e apidoc,docs

.. warning:: Building the docs requires ``graphviz``, in addition to what Tox_ can manage.

   If you see::

      WARNING: dot command 'dot' cannot be run (needed for graphviz output), check the graphviz_dot setting

   Then execute::

      sudo apt install graphviz

Documentation looks pre-CSS? If you see::

   copying static files... WARNING: cannot copy static file NotADirectoryError(20, 'Not a directory')

Then execute::

   rm dist/docs/_static
   tox -e docs

``dist/docs/_static`` is supposed to be a directory but sometimes when building from clean state,
it might just be one file of that directory.

Packaging
---------

Before building, you should clean the building area::

    rm -rf build
    rm -rf src/*.egg-info

Then, make sure that everything is in order::

    python3 -m tox -e check

Build the ``sdist`` (and ``bdist_wheel``)::

    python3 setup.py clean --all sdist bdist_wheel

You should now have a new release in ``dist/``!

.. _Tox: https://tox.readthedocs.io/en/latest/
.. _Sphinx: http://sphinx-doc.org/
.. _Setuptools: https://pypi.python.org/pypi/setuptools
.. _Pytest: http://pytest.org/
.. _isort: https://pypi.python.org/pypi/isort
