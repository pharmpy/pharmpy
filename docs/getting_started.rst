===============
Getting started
===============

Pharmpy can be used in a python program as a module or via its command line interface. It can also
be used via reticulate from R.

The :class:`pharmpy.Model` class is representation of a nonlinear mixed effects model. For example, to
read the NONMEM model ``pheno_real.mod`` from file into a model object, the following is sufficient:

.. testcode::
    :pyversion: > 3.6

    from pharmpy import Model

    path = 'tests/testdata/nonmem/pheno_real.mod'
    pheno = Model(path)

The model file format is automatically detected:

.. doctest::
    :pyversion: > 3.6

    >>> type(pheno)
    <class 'pharmpy.plugins.nonmem.model.Model'>

------
From R
------

To call Pharmpy from R the following is needed on your computer:

#. R
#. The reticulate R package
#. Python 3.6 or newer
#. Pharmpy

Here is an example of how to use Pharmpy from R:

.. code-block:: R

    library(reticulate)
    use_python("python3")     # Only needed if your python interpreter is not in the path or
                              # has a non-standard name

    pharmpy <- import("pharmpy")
    model <- pharmpy$Model("run1.mod")
    params <- model$parameters
    params$inits <- list('THETA(1)'=2)  # set initial estimate of THETA(1) to 2
    model$parameters <- params
    model$write("run2.mod")
