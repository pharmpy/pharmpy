===============
Getting started
===============

Pharmpy can be used in a Python program as a module or via its command line interface. It can also
be used via reticulate from R.

------------
Installation
------------

.. warning:: Pharmpy requires python 3.6 or later,
    and is currently tested on python 3.6, 3.7, 3.8 and 3.9 on Linux, MacOS and Windows.

Install the latest stable version from PyPI with::

   pip install pharmpy-core

---------------
A first example
---------------

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

For examples of how the Pharmpy model works and how you can transform it, see :ref:`here <model>` and
:ref:`here <modeling>`.

------
From R
------

Pharmpy can also be used in R. To call Pharmpy from R the following is needed on your computer:

#. The reticulate R package
#. Python 3.6 or newer
#. Pharmpy

See :ref:`here <using_r>` for more information on how to use Pharmpy in R.
