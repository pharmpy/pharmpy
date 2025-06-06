===============
Getting started
===============

Pharmpy can be used in a Python program as a library or via its command line interface. It can also
be used via its R wrapper package: `pharmr <https://github.com/pharmpy/pharmr>`_.

-------------------
Supported platforms
-------------------

+------------------+---------------+
| Operating system | Architecture  |
+==================+===============+
| Linux            | x64           |
+------------------+---------------+
|                  | aarch64       |
+------------------+---------------+
| MacOS            | x64           |
+------------------+---------------+
|                  | aarch64       |
+------------------+---------------+
| Windows          | x64           |
+------------------+---------------+

------------
Installation
------------

.. note::
    If you plan to use Pharmpy in R, please follow the steps in :ref:`install_pharmr`.

.. warning::
    Pharmpy requires Python 3.11 or later, and is currently tested on Python 3.11, 3.12 and 3.13
    in all supported platforms.

Install the latest stable version from PyPI with::

   pip install pharmpy-core

To be able to use components using machine learning the tflite package is needed. For general usage of Pharmpy or pharmr this is not needed. It can be installed using::

    pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtime

---------------
A first example
---------------

The :class:`pharmpy.model.Model` class is representation of a nonlinear mixed effects model. For example, to
load an example NONMEM model:

.. pharmpy-execute::
   :hide-output:

    from pharmpy.modeling import load_example_model, print_model_code

    model = load_example_model('pheno')
    print_model_code(model)

The model file format is automatically detected:

.. pharmpy-execute::

    type(model)

You can read more about how to read in and examine your model in :ref:`model`, and how to transform your model via
:ref:`modeling`. We also have several :ref:`pharmpy_tools` to run more complex workflows.

.. note::

    In order to run any of the tools you need to have a configuration file set up with a path to NONMEM, instructions
    can be found :ref:`here <config_page>`.

========
Examples
========

Examples can be found here

.. toctree::
   :maxdepth: 1

   Simple estimation example <getting_started_example_1>
