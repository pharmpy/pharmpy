.. _using_r:

============
Pharmpy in R
============

`pharmr <https://github.com/pharmpy/pharmr>`_ is an R wrapper to Pharmpy. It provides an R interface to all functions
found in the modeling-module (documented `here <https://pharmpy.github.io/latest/reference/pharmpy.modeling.html>`_).
Each function is also available via the help-function (or `?`). The version number of pharmr
mirrors Pharmpy, so it is import to make sure they have the same version number (a warning
will appear if they are different). pharmr version 0.33.1 is on CRAN. However, since Pharmpy
still is under development and different features constantly are being released, we still
recommend using the development version on Github until Pharmpy/pharmr version 1.0.0 is released.

Using pharmr
~~~~~~~~~~~~

In pharmr, you can pipe different Pharmpy functions together with the magrittr-package:

.. code-block:: r

    library(pharmr)
    model <- load_example_model('pheno') %>%
      add_parameter('MAT') %>%
      fit()

.. note::

    If you try to access data frames belonging to a Pharmpy object you need to reset the index.
    All functions available in pharmr do this internally, it is only when you have data frames
    nested in objects (such as a model object) that you need to do this. If we continue the previous
    example:

    .. code-block:: r

        residuals <- reset_index(model$modelfit_results$residuals)

Gotchas
~~~~~~~

The model object
================

In Pharmpy, all changes to a model object occur in place.

.. code-block:: r

    model_a <- model_b <- load_example_model(’pheno’)

All changes to model_a will be also applied to model_b since they refer to the same object.
In order to have two different models, you can do the following:

.. code-block:: r

    model_a <- load_example_model(’pheno’)
    model_b <- copy_model(model_a, name=’pheno2’)

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
