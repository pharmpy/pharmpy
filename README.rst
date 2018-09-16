.. highlight:: console

========
Overview
========

.. start-longdesc

PharmPy is an experiment in substituting (some of) the PsN functionality in Python.

For PsN (implemented in Perl), see `UUPharmacometrics/PsN
<https://github.com/UUPharmacometrics/PsN/releases>`_.

.. end-longdesc

Installation
============

See :ref:`installation-section`.

Documentation
=============

See local documentation in ``dist/docs/``

.. warning:: Building the docs requires ``graphviz``, in addition to what Tox_ can manage.

   If you see::

      WARNING: dot command 'dot' cannot be run (needed for graphviz output), check the graphviz_dot setting

   Then execute::

      sudo apt install graphviz

.. warning:: Documentation look like a pre-CSS site?

   If you see::

      copying static files... WARNING: cannot copy static file NotADirectoryError(20, 'Not a directory')

   Then execute::

      rm dist/docs/_static

   And re-build::

      tox -e docs

   ``dist/docs/_static`` is supposed to be a directory but sometimes when building from clean state,
   it might just be one file of that directory... Why?

Development
===========

Testing
-------

To run the all tests via Pytest_ install Tox_::

    pip3 install tox

Then run::

    python3 -m tox

Note, to combine the coverage data from all the Tox_ environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox

Missing a Python interpreter? Ubuntu 18.04 and no more `python3.5`? No worries! Do this::

    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install python3.5-dev

Don't worry. Everything will be prefixed `python3.5` so there won't be any collision with e.g.
`python3` (it'll still use the system standard).

Building
--------

This project uses `Semantic Versioning 2.0.0 <https://semver.org/>`_, and
has a bumpversion_ config in ``.bumpversion.cfg``. Thus, remember to run:

* ``bumpversion patch`` to increase version from `0.1.0` to `0.1.1`.
* ``bumpversion minor`` to increase version from `0.1.0` to `0.2.0`.
* ``bumpversion major`` to increase version from `0.1.0` to `1.0.0`.

Before building, you should clean the building area::

    rm -rf build
    rm -rf src/*.egg-info

Then, make sure that everything is in order::

    python3 -m tox -e check

Build the ``sdist`` (and ``bdist_wheel``)::

    python3 setup.py clean --all sdist bdist_wheel

You should now have a new release in ``dist/``!

.. _Tox: https://tox.readthedocs.io/en/latest/
.. _Sphinx: http://sphinx-doc.org/
.. _Setuptools: https://pypi.python.org/pypi/setuptools
.. _Pytest: http://pytest.org/
.. _isort: https://pypi.python.org/pypi/isort
.. _bumpversion: https://pypi.org/project/bumpversion
