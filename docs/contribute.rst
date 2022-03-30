==========
Contribute
==========

We encourage your contribution to the Pharmpy project! Anything from plugins for new model types to reporting bugs as
GitHub issues, any contribution is appreciated. See also :ref:`projects` for project suggestions for studens or interns.

Set up of development environment
#################################

Install and run tox
*******************

Pharmpy is built and tested using tox. Install it with::

    pip3 install tox

To run the entire workflow simply run::

    tox


Check code formatting
*********************

Pharmpy use the black code formatter, the flake8 linter and isort for formatting of imports. Formatting and linting can be run separately with::

    tox -e check


Run the unit tests
******************

To run all unit tests for one python version::

    tox -e py39

To run a particular unit test file::

    tox -e py39 :: pytest test_parameters.py

To run a particular test function in a particular test file::

    tox -e py39 :: pytest test_parameters.py -ktest_myfunc

To run a particular test function in a particular integration test file::

    tox -e integration :: pytest tests/integration/test_modelsearch.py -k test_summary_individuals


Build
*****

To build a usable virtual environment without having to run all tests:

    tox -e run -- pharmpy --version


Build the documentation
***********************

To build the documentation separately run:

    tox -e docs

The local version of the documentation can now be found in ``dist/docs/``.


Install dependencies for generating documentation
*************************************************

Graphviz is needed to render various graphs in the documentation. This can be installed via
the following commands.

On Ubuntu:

.. code-block:: sh

    sudo apt install graphviz

On Mac:

.. code-block:: sh

    brew install graphviz

For alternative installations or for Windows, please see Graphviz `guide <https://graphviz.org/download/>`_

Releasing Pharmpy and documentation
###################################

A deployment workflow is setup on github actions to automatically release Pharmpy to PyPI and the documentation to pharmpy.github.io. The workflow also triggers an automatic update of pharmr. This workflow will be triggered when pushing a version tag, i.e. a tag starting with 'v'.

1. Update the CHANGELOG
2. Bump the version (see below)
3. `git push`
4. `git push --tags`

Version
*******

Pharmpy uses `Semantic Versioning 2.0.0 <https://semver.org/>`_, and
has a bumpversion_ config in ``.bumpversion.cfg``. Run the wrapper

* ``./bumpversion.sh patch`` to increase version from `0.1.0` to `0.1.1`.
* ``./bumpversion.sh minor`` to increase version from `0.1.0` to `0.2.0`.
* ``./bumpversion.sh major`` to increase version from `0.1.0` to `1.0.0`.

This will also update the year of copyright to the current year if necessary.

.. _bumpversion: https://pypi.org/project/bumpversion

Information for developers
##########################

.. toctree::
   :maxdepth: 2

   Design principles <design>
   NONMEM
