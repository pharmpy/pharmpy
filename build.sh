rm dist/*.whl dist/*.tar.gz

python3 setup.py bdist_wheel
python3 setup.py sdist --formats=gztar

twine upload dist/*.whl dist/*.tar.gz
