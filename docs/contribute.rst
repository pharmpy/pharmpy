.. _contribute:

==========
Contribute
==========

We encourage your contribution to the Pharmpy project! Anything from plugins for new model types to reporting bugs as
GitHub issues (to `Pharmpy <https://github.com/pharmpy/pharmpy/issues>`_ or to `pharmr <https://github.com/pharmpy/pharmr/issues>`_),
any contribution is appreciated. See also :ref:`projects` for project suggestions. These can be suitable for students or interns.
Our community aspires to treat everyone equally and to value all contributions. We have a :doc:`Code of Conduct <code_of_conduct>`
to foster an open and welcoming environment.

Set up the development environment
##################################

Supported platforms
*******************

Pharmpy supports Linux, MacOS and Windows and it is possible for developers to use any of these platforms.

Install Python
**************

Some operating systems come with Python already installed, but on others you'll need to install it. Check the badge on the Pharmpy github page for a list of supported Python versions. Python and pip are usually called with the ``python`` and ``pip`` commands, but on some platforms ``python3`` and ``pip3`` have to be used. In this guide we'll use the former.

Install git
***********

Pharmpy is using git for source control. It can for example be installed on Ubuntu with::

    apt install git

Fork and clone the code repository
**********************************

Fork the Pharmpy repository at https://github.com/pharmpy/pharmpy
Clone the fork to your computer::

    git clone <url of fork>

Install tox
***********

Pharmpy is built and tested using tox. Install it with::

    pip install tox

Install graphviz used for building the documentation
****************************************************

Graphviz is needed to render various graphs in the documentation. This can be installed via
the following commands.

On Ubuntu:

.. code-block:: sh

    sudo apt install graphviz

On Mac:

.. code-block:: sh

    brew install graphviz

For alternative installations or for Windows, please see Graphviz `guide <https://www.graphviz.org/download/>`_

Using the development environment
#################################

Running tox
***********

To run the entire workflow simply run::

    tox

This will run code checking, unit testing, building of documentation and documentation testing. Often it is unnecessary to
run all these things while developing so below are instructions on how to run parts separately.

Check code formatting
*********************

Pharmpy use the ``black`` code formatter, the ``flake8`` linter and ``isort`` for formatting of imports. Formatting and linting can be run separately with::

    tox -e format

Developers can format the code any way they want while writing code and the formatter will automatically reformat.

Run the unit tests
******************

To run all unit tests for one python version::

    tox -e py39

To run a particular unit test file::

    tox -e py39 -- tests/modeling/test_parameters.py

To run a particular test function in a particular test file::

    tox -e py39 -- tests/modeling/test_parameters.py::test_get_thetas

Build and test the documentation
********************************

The documentation can be built with::

    tox -e docs-build

The local version of the documentation can now be found in ``dist/docs/``.

Documentation tests are run with::

    tox -e docs-test

This will test all code given in examples of function documentation and check their output.

Run the integration tests
*************************

Apart from the unit tests Pharmpy also has a suite of integration tests. The purpose of these
are to test complex workflows together with external tools, e.g. NONMEM and nlmixr. To run these
you will need to have a version of NONMEM installed.

To run all integration tests::

    tox -e integration

To run a particular test function in a particular integration test file::

    tox -e integration -- tests/integration/test_modelsearch.py -k test_summary_individuals

Build a usable virtual environment
**********************************

To build a usable virtual environment without having to run all tests::

    tox -e plain -- pharmpy --version

This will create a plain virtual environment in the ``.tox/run`` directory that can be used for using Pharmpy.

Standard workflow for contributing
##################################

Pharmpy is using github to centralize the development. Issues, pull requests and continuous integration tests are managed at github.

Contributors should employ the standard pull request workflow. The steps in short:

1. Create a local branch for the feature you will be working on
2. Write the code and unit tests
3. Run ``tox -e format``
4. Commit changes
5. Push the branch to the fork: ``git push origin <name_of_branch>``
6. Create a pull request in the Pharmpy github page
7. Tests will now be run automatically on github actions for all supported platforms

Releasing Pharmpy and documentation
###################################

.. note::
    This information is for maintainers.

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

   NONMEM
