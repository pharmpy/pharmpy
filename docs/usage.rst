=====
Usage
=====

The :class:`pharmpy.Model` class is representation of a nonlinear mixed effects model. For example, to
read the NONMEM model ``pheno_real.mod`` from file into a model object, the following is sufficient:

.. testcode::
    :pyversion: > 3.6

    from pharmpy import Model

    path = 'tests/testdata/nonmem/pheno_real.mod'
    pheno = Model(path)

The API is automatically detected and used:

.. doctest::
    :pyversion: > 3.6

    >>> type(pheno)
    <class 'pharmpy.plugins.nonmem.model.Model'>
