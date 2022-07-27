===============
Getting started
===============

Pharmpy can be used in a Python program as a library or via its command line interface. It can also
be used via its R wrapper package: `pharmr <https://github.com/pharmpy/pharmr>`_.

------------
Installation
------------

.. note::
    If you plan to use Pharmpy in R, please follow the steps in :ref:`install_pharmr`.

.. warning::
    Pharmpy requires python 3.8 or later, and is currently tested on python 3.8, 3.9, and 3.10 on
    Linux, MacOS and Windows.

Install the latest stable version from PyPI with::

   pip install pharmpy-core

To be able to use components using machine learning the tflite package is needed. It can be installed using::

    pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtime

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

You can read more about how to read in an examine your model in :ref:`model`, and how to transform your model via
:ref:`modeling`. We also have several :ref:`pharmpy_tools` to run more complex workflows.


