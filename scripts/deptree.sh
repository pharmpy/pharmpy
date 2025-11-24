#!/bin/bash

# Print the dependency tree of Pharmpy 
# Actual dependencies will be used (i.e. not requirements.txt)

rm -rf /tmp/updatevenv
python3 -m venv /tmp/updatevenv
activate () {
    . /tmp/updatevenv/bin/activate
}
activate
pip install setuptools --upgrade
cd ..
pip install .
pip install pipdeptree
pipdeptree -fl

exit
