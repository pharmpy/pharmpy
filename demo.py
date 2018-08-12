#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import importlib
import sys
from os import listdir
from os.path import dirname
from os.path import join
from os.path import realpath


def iprint(text, width=4):
    if isinstance(text, str):
        lines = text.splitlines()
    else:
        lines = ['%d : %s' % (i, str(x)) for i, x in enumerate(text)]
    lines = [' '*width + line for line in lines]
    print('\n'.join(lines))


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
    iprint(str(model))

    print('\nmodel.input.path = %s' % (model.input.path,))
    names = model.input.column_names()
    print('\nmodel.input.columns_names() =')
    iprint(names)

    thetas = model.get_records('THETA')
    print("\nmodel.get_records('THETA') =")
    iprint(thetas)

    print("\nmodel.get_records('THETA')[0].lexer =")
    print(thetas[0].lexer)

# lexer.test()
