.. _using_r:

============
Pharmpy in R
============

`pharmr <https://github.com/pharmpy/pharmr>`_ is an R wrapper to Pharmpy. It provides an R interface to all functions
found in the modeling-module (documented `here <https://pharmpy.github.io/latest/reference/pharmpy.modeling.html>`_).
Each function is also available via the help-function (or ``?``). The version number of pharmr mirrors Pharmpy, so it is
import to make sure they have the same version number (a warning will appear if they are different). pharmr is available on
CRAN. However, since Pharmpy still is under development and different features constantly are being
released, we still recommend using the development version on Github until Pharmpy/pharmr version 1.0.0 has been released.

.. _install_pharmr:

Installing pharmr
~~~~~~~~~~~~~~~~~

pharmr uses the package `reticulate <https://rstudio.github.io/reticulate>`_ for calling Python from R. Install
pharmr and Pharmpy with the following:

.. code-block:: r

    remotes::install_github("pharmpy/pharmr", ref="main")
    pharmr::install_pharmpy()

It is also possible to install from CRAN (but we still recommend using the GitHub installation until 1.0.0 is out):

.. code-block:: r

    install.packages("pharmr")
    pharmr::install_pharmpy()

.. note::
    Make sure to clear your environment and restart the R session after installing/updating.

To make sure the versions of Pharmpy and pharmr are up-to-date and were installed successfully, run the following:

.. code-block:: r

    library(pharmr)
    packageVersion("pharmr")
    print_pharmpy_version()

If you get any error during the installation or when running the code above, please check :ref:`trouble_shoot_pharmr`.

.. _trouble_shoot_pharmr:

Trouble shooting
================

Wrong Python version
--------------------

When reticulate sets up Miniconda it can default to use Python 3.6 (which Pharmpy does not support). If you have any
trouble installing Pharmpy or any of its dependencies, you can do the following to check the Python version in your
reticulate environment:

.. code-block:: r

    reticulate::py_discover_config()

Make sure the Python version is >= 3.8. If it is not, you can run the following in R:

.. code-block:: r

    reticulate::conda_create('r-reticulate', python_version = '3.9')

Restart the session and try installing Pharmpy again:

.. code-block:: r

    pharmr::install_pharmpy()

Error importing Pharmpy
-----------------------

If you have followed the installation steps but still get an error asking you to install Pharmpy, the issue might be
that pharmr is not using or finding the right virtual environment. If you used pharmr to set up your environment
(default when you use ``install_pharmpy()``), run the following:

.. code-block:: r

    reticulate::py_discover_config()

and make sure you have something similar to the following as the output:

.. code-block::

    python:         .../r-reticulate/bin/python
    libpython:      .../r-reticulate/lib/libpython3.10.so
    ...

Using pharmr
~~~~~~~~~~~~

In pharmr, you can pipe different Pharmpy functions together with the magrittr-package:

.. code-block:: r

    library(pharmr)
    library(magrittr)
    model <- read_model('path/to/model') %>%
      set_zero_order_absorption() %>%
      fit()

Gotchas
~~~~~~~

List indices
============

One difference between Python and R is that in Python, list indices start at 0 while in R
it starts at 1. Since Pharmpy is developed in Python, in functions where you have arguments
referring to indices, you need to use the Python way. For example:

.. code-block:: r

    set_estimation_step(model, method, interaction = TRUE, options = NULL, idx = 0)

Note that normal R data structures such as vectors, lists and data frames are still indexed
the same way as usual:

.. code-block:: r

    etas <- model$random_variables
    etas[1] # access first element
