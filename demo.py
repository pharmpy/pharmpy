#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import importlib
import sys
from pprint import pprint
from os import listdir
from os.path import dirname
from os.path import join
from os.path import realpath


def pprint_str(obj, *args, **kwargs):
    kwargs['indent'] = kwargs.get('indent', 2)
    if isinstance(obj, str):
        out = obj
    else:
        try:
            out = list(map(lambda x: x if isinstance(x, str) else str(x), obj))
        except TypeError:
            out = obj
    pprint(out, *args, **kwargs)


# load ./src/pysn as library 'pysn'
root = dirname(realpath(__file__))
sys.path.append(join(root, 'src'))
pysn = importlib.import_module('pysn')
# lexer = importlib.import_module('pysn.psn.lexer')
sys.path.pop()

path = join(root, 'tests', 'testdata', 'nonmem')
for path_model in [join(path, file) for file in listdir(path)]:
    model = pysn.Model(path_model)
    print("pysn.Model('%s')" % path_model)
    print('='*80)

    print('str(model) =')
    pprint_str(str(model))

    print('\nmodel.input.path = %s' % (model.input.path,))
    names = model.input.column_names()
    print('\nmodel.input.columns_names() =')
    pprint_str(names)

    print("recs = model.get_records('THETA')")
    recs = model.get_records('THETA')
    for i, rec in enumerate(recs):
        for j, theta in enumerate(rec.thetas):
            print('recs[%d].thetas[%d] = %s' % (i, j, theta))

    print("\nmodel.get_records('THETA')[0].parser =")
    print(recs[0].parser)

# lexer.test()
