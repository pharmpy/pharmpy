.. _using_r:

============
Pharmpy in R
============

Using Pharmpy in R is similar to how it is used in Python, and all the examples for Python are analogous to how
they work in R. Replace "." with "$" for the R equivalent. When a list (denoted [...] in Python) is used as input
in e.g. transformations, the R equivalent is `c()`. For more information about type conversions from R to Python,
see `Reticulate's documentation <https://rstudio.github.io/reticulate/index.html#type-conversions>`_.

An R wrapper to Pharmpy is in development.

.. warning::

    When providing a one-element list as an input argument to e.g. transformations or configurations that do not
    support single strings, use `list()` instead of `c()`. This is because Reticulate translates the string array
    from the list element to separate strings (e.g. `c('ETA(1)')` will be interpreted as
    `c('E', 'T', 'A', '(', '1', ')')`.

If you want the documentation for a function, use the function `py_help()` from Reticulate.

----------------
Relevant imports
----------------
.. code-block:: r

    >>> library(reticulate)
    >>> # use_python("python3")  # Uncomment if needed
    >>> pharmpy <- import("pharmpy")

A way to write functions from e.g. the modeling module in a more clean way:

.. code-block:: r

    >>> add_covariate_effect <- modeling$add_covariate_effect

----------------
Basic operations
----------------
To load a model and access e.g. parameters

.. code-block:: r

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
    >>> model$write("run2.mod")

---------------
Transformations
---------------
It is also possible to perform different types of transformations, such as addition of covariate effects (see the
user guide :ref:`Modeling <modeling>` for available transformations).

Together with the `dplyr` package, you can create transformation pipelines (note that model is changed in place).

.. code-block:: r

    >>> model <- pharmpy$Model("run1.mod") %>%
    >>>     first_order_absorption() %>%
    >>>     add_iiv("MAT", "exp") %>%
    >>>     update_source() %>%
    >>>     write_model("run2.mod")


