==========
Contribute
==========

.. warning::
   This information is mostly outdated and will be updated shortly.

.. toctree::
   :maxdepth: 4

   NONMEM
   todo
   version
   design

Development Environment Installation
####################################

Multiple Versions of Python
***************************
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

    pyenv install 3.6.10
    pyenv install 3.7.7
    pyenv install 3.8.3

5. Change to folder where tests are run (in this case pharmpy) and set local python versions

.. code-block:: sh

    pyenv local 3.6.10 3.7.7 3.8.3

Sources:
https://realpython.com/intro-to-pyenv/
https://github.com/pyenv/pyenv

Rendering model compartments
****************************
In order to render a visualization of the compartment graphs you need to install Graphviz. This can be done via
the following commands.

On Ubuntu:

.. code-block:: sh

    sudo apt install graphviz

On Mac:

.. code-block:: sh

    brew install graphviz

For alternative installations or for Windows, please see Graphviz `guide <https://graphviz.org/download/>`_
