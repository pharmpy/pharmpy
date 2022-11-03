#!/bin/bash

# This script will create a venv, install python and do pip freeze
# (note that it doesn't work with virtualenv since it is not clean after installation)
# The freeze will be compared with the current requirements.txt
# New packages will be added, removed will be removed all others will be kept
# with the versions in requirements.txt

rm -rf /tmp/updatevenv
python3 -m venv /tmp/updatevenv
activate () {
    . /tmp/updatevenv/bin/activate
}
activate
pip install setuptools --upgrade
cd ..
#python3 setup.py install
pip install .
pip freeze >/tmp/updatevenv/freeze
python - << EOF
with open("/tmp/updatevenv/freeze", "r") as f1, open("requirements.txt", "r") as f2, open("/tmp/newreq", "w") as dh:
    d = {}
    keep = []
    for line in f2:
        a = line.split('==')
        d[a[0]] = line
        if line.startswith("tflite") or line.startswith("--extra-index-url"):
            keep.append(line)
    for line in f1:
        a = line.split('==')
        if a[0] == 'pharmpy-core':
            pass
        elif a[0] in d:
            print(d[a[0]], file=dh, end="")
        else:
            print(line, file=dh, end="")
    for line in keep:
        print(line, file=dh, end="")
EOF

cp /tmp/newreq requirements.txt

exit
