========
Overview
========

.. start-summary

PysN is an experiment in substituting (some of) the PsN functionality in python

.. end-summary

Installation
============

::

    pip install -e build/lib/pysn

Documentation
=============

See `docs/index.html`

Development
===========

To run the all tests install `tox <https://tox.readthedocs.io>`_::

    pip3 install tox

Then run::

    python3 -m tox

Note, to combine the coverage data from all the `tox` environments run:

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
