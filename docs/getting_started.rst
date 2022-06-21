===============
Getting started
===============

Pharmpy can be used in a Python program as a library or via its command line interface. It can also
be used via its R wrapper package: `pharmr <https://github.com/pharmpy/pharmr>`_.

------------
Installation
------------

In Python: Pharmpy
~~~~~~~~~~~~~~~~~~

.. warning::
    Pharmpy requires python 3.8 or later, and is currently tested on python 3.8, 3.9, and 3.10 on
    Linux, MacOS and Windows.

Install the latest stable version from PyPI with::

   pip install pharmpy-core

To be able to use components using machine learning the tflite package is needed. It can be installed using::

    pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtime



In R: pharmr
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

.. note::
    For more information and gotchas of using pharmr, see :ref:`Using R<using_r>`.

---------------
A first example
---------------

The :class:`pharmpy.Model` class is representation of a nonlinear mixed effects model. For example, to
read an example NONMEM model (in this case ``pheno``):

.. pharmpy-execute::
   :hide-output:

    from pharmpy.modeling import load_example_model, print_model_code

    model = load_example_model('pheno')
    print_model_code(model)

The model file format is automatically detected:

.. pharmpy-execute::

    type(model)

For examples of how the Pharmpy model works and how you can transform it, see :ref:`here <model>` and
:ref:`here <modeling>`.


.. pharmpy-code::

    from pharmpy.modeling import read_model

    path = 'path/to/model'
    pheno = read_model(path)

