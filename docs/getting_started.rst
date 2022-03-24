===============
Getting started
===============

Pharmpy can be used in a Python program as a library or via its command line interface. It can also
be used via its R wrapper package: `pharmr <https://github.com/pharmpy/pharmr>`_.

--------------------------
Getting started in Pharmpy
--------------------------

Installation
~~~~~~~~~~~~

.. warning:: Pharmpy requires python 3.8 or later,
    and is currently tested on python 3.8 and 3.9 on Linux, MacOS and Windows.

Install the latest stable version from PyPI with::

   pip install pharmpy-core

To be able to use components using machine learning the tflite package is needed. It can be installed using::

    pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtime


A first example
~~~~~~~~~~~~~~~

The :class:`pharmpy.Model` class is representation of a nonlinear mixed effects model. For example, to
read the NONMEM model ``pheno_real.mod`` from file into a model object, the following is sufficient:

.. testcode::
    :pyversion: > 3.7

    from pharmpy.modeling import read_model

    path = 'tests/testdata/nonmem/pheno_real.mod'
    pheno = read_model(path)

The model file format is automatically detected:

.. doctest::
    :pyversion: > 3.7

    >>> type(pheno)
    <class 'pharmpy.plugins.nonmem.model.Model'>

For examples of how the Pharmpy model works and how you can transform it, see :ref:`here <model>` and
:ref:`here <modeling>`.

-------------------------
Getting started in pharmr
-------------------------

Installation
~~~~~~~~~~~~

pharmr uses the package `reticulate <https://rstudio.github.io/reticulate>`_ for calling Python from R. Install
pharmr and Pharmpy with the following:

.. code-block:: r

    remotes::install_github("pharmpy/pharmr", ref="main")
    pharmr::install_pharmpy()

Trouble shooting
================

When reticulate sets up Miniconda it can default to use Python 3.6 (which Pharmpy does not
support). If you have any trouble installing Pharmpy or any of its dependencies, you can do
the following to check the Python version in your reticulate environment:

.. code-block:: r

    library(reticulate)
    reticulate::py_discover_config()

Make sure the Pyrhon version is >= 3.8. If it is not, you can run the following in R:

.. code-block:: r

    conda_create('r-reticulate', python_version = '3.9')

Restart the session and try installing Pharmpy again:

.. code-block:: r

    library(pharmr)
    pharmr::install_pharmpy()

A first example
~~~~~~~~~~~~~~~

Using the same example as in the Pharmpy example:

.. code-block:: r

    library(pharmr)

    path <- 'tests/testdata/nonmem/pheno_real.mod'
    pheno <- read_model(path)

For more information and gotchas of using pharmr, see :ref:`Using R<using_r>`.
