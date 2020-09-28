=======
Using R
=======

Using Pharmpy in R is simple, and all the examples for Python are analogous to how they work in R. Simply replace "."
with "$" for the R equivalent.

Relevant imports:

.. code-block:: r

    >>> library(reticulate)
    >>> use_python("python3")
    >>> pharmpy <- import("pharmpy")

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

It is also possible to perform different types of transformations, such as addition of covariate effects.
