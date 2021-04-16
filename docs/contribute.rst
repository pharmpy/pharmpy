==========
Contribute
==========

We encourage your contribution to the Pharmpy project! Anything from plugins for new model types to reporting bugs as
GitHub issues, any contribution is appreciated.

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


Build the documentation
***********************

To build the documentation separately run:

    tox -e apidoc,docs

The local version of the documentation can now be found in ``dist/docs/``.


Install Multiple Versions of Python for Testing
***********************************************

On Ubuntu:

1. Install dependencies

.. code-block:: sh

    sudo apt-get update
    sudo apt-get install --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

2. Clone repository/install pyenv

.. code-block:: sh

    git clone https://github.com/pyenv/pyenv.git ~/.pyenv

OR

.. code-block:: sh

    curl https://pyenv.run | bash

3. Add to .bashrc so that pyenv is loaded automatically

.. code-block:: sh

    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc

4. Install python versions of interest

.. code-block:: sh

    pyenv install 3.7.7
    pyenv install 3.8.3
    pyenv install 3.9.1

5. Change to folder where tests are run (in this case pharmpy) and set local python versions

.. code-block:: sh

    pyenv local 3.7.7 3.8.3 3.9.1

Sources:
https://realpython.com/intro-to-pyenv/
https://github.com/pyenv/pyenv

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

A deployment workflow is setup on github actions to automatically release Pharmpy to PyPI and the documentation to pharmpy.github.io. This workflow will be triggered when pushing a version tag, i.e. a tag starting with 'v'.

1. Check that the CHANGELOG has been updated
2. Bump the version (see Version below)
3. `git push`
4. `git push --tags`

Version
#######

Pharmpy uses `Semantic Versioning 2.0.0 <https://semver.org/>`_, and
has a bumpversion_ config in ``.bumpversion.cfg``. Thus, remember to run:

* ``bumpversion patch`` to increase version from `0.1.0` to `0.1.1`.
* ``bumpversion minor`` to increase version from `0.1.0` to `0.2.0`.
* ``bumpversion major`` to increase version from `0.1.0` to `1.0.0`.

.. _bumpversion: https://pypi.org/project/bumpversion

Further reading
###############

.. toctree::
   :maxdepth: 2

   NONMEM
   todo
   design
