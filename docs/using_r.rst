.. _using_r:

============
Pharmpy in R
============

`Pharmr <https://github.com/pharmpy/pharmr>`_ is an R wrapper to Pharmpy. It provides an R interface to all functions
found in the modeling-module (documented `here <https://pharmpy.github.io/latest/reference/pharmpy.modeling.html>`_).
Each function is also available via the help-function (or `?`). The version number of Pharmr
mirrors Pharmpy, so it is import to make sure they have the same version number (a warning
will appear if they are different). Pharmr version 0.33.1 is on CRAN. However, since Pharmpy
still is under development and different features constantly are being released, we still
recommend using the development version on Github until Pharmpy/Pharmr version 1.0.0 is released.

Installation
~~~~~~~~~~~~
Pharmr uses the package `reticulate <https://rstudio.github.io/reticulate>`_ for calling Python from R. When installing Pharmr,
reticulate will give a prompt to set up the reticulate environment. To use Python 3.9
in your environment run the following code in R before installing Pharmr:

.. code-block:: r

    Sys.setenv(RETICULATE_MINICONDA_PYTHON_VERSION="3.9")

Then install Pharmr and Pharmpy:

.. code-block:: r

    remotes::install_github("pharmpy/pharmr", ref="main")
    pharmr::install_pharmpy()

Trouble shooting
================
When reticulate sets up Miniconda it will default to use Python 3.6 (which Pharmpy does not
support). If you have any trouble installing Pharmpy or any of its dependencies, you can do
the following to check the Python version in your reticulate environment:

.. code-block:: r

    library(reticulate)
    reticulate::py_discover_config()

Make sure the Pyrhon version is >= 3.7. If it is not, you can run the following in R:

.. code-block:: r

    conda_create('r-reticulate', python_version = '3.9')

Restart the session and try installing Pharmpy again:

.. code-block:: r

    library(pharmr)
    pharmr::install_pharmpy()

Using Pharmr
~~~~~~~~~~~~
In Pharmr, you can pipe different Pharmpy functions together with the magrittr-package:

.. code-block:: r

    library(pharmr)
    model <- load_example_model('pheno') %>%
      add_parameter('MAT') %>%
      fit()

`fit() <https://pharmpy.github.io/latest/modelfit.html>`_ will call the appropriate software/package to run the model (e.g. a NONMEM model will
be run using NONMEM, currently only this is supported). Most methods return model object
references, allowing you to pipe more easily.

.. note::

    If you try to access data frames belonging to a Pharmpy object you need to reset the index.
    All functions available in Pharmr do this internally, it is only when you have data frames
    nested in objects (such as a model object) that you need to do this. An example:

    .. code-block:: r

        model <- load_example_model('pheno')
        residuals <- reset_index(model$modelfit_results$residuals)

Gotchas
~~~~~~~
The model object
================
In Pharmpy, all changes to a model object occur in place.

.. code-block:: r

    model_a <- model_b <- load_example_model(’pheno’)

All changes to model_a will be also applied to model_b since they refer to the same object.
In order to have to different models we recommend the following to have two different models:

.. code-block:: r

    model_a <- load_example_model(’pheno’)
    model_b <- copy_model(model_a, name=’pheno2’)

List indices
============
One difference between Python and R is that in Python, list indices start at 0 while in R
it starts at 1. Since Pharmpy is developed in R, in functions where you have arguments
referring to indices, you need to use the Python way. For example:

.. code-block:: r

    set_estimation_step(model, method, interaction = TRUE, options = NULL, idx = 0)

Note that normal R data structures such as vectors, lists and data frames are still indexed
the same way as usual:

.. code-block:: r

    etas <- model$random_variables
    etas[1] # access first element
