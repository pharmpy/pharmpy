#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import importlib
import sys
from os.path import dirname
from os.path import join
from os.path import realpath
from pprint import pprint

import pandas as pd


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
for path_model in [join(path, file) for file in ('pheno_real.mod',)]:
    model = pysn.Model(path_model)
    print("pysn.Model('%s')" % path_model)
    print('='*80)

    print('str(model) =')
    pprint_str(str(model))

    print("recs = model.get_records('THETA')")
    recs = model.get_records('THETA')
    for i, rec in enumerate(recs):
        for j, theta in enumerate(rec.thetas):
            print('recs[%d].thetas[%d] = %s' % (i, j, theta))

    print("\nmodel.get_records('THETA')[0].parser =")
    print(recs[0].parser)

    print("\nmodel.input.data_frame =")
    pd.set_option('display.max_rows', 10)
    print(model.input.data_frame)

    print("\nmodel.parameters.population:")
    for i, pop in enumerate(model.parameters.population):
        print('  %d %s' % (i, pop))

    print("\nmodel.get_records('OMEGA')[0].block = ")
    mat = model.get_records('OMEGA')[0].block
    print(mat)

# lexer.test()
