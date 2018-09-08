#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import importlib
import sys
from pathlib import Path
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
root = Path(__file__).resolve().parent
sys.path.append(str(root / 'src'))
pysn = importlib.import_module('pysn')
# lexer = importlib.import_module('pysn.psn.lexer')
sys.path.pop()

path = root / 'tests' / 'testdata' / 'nonmem'
for mpath in [path / file for file in ('pheno_real.mod',)]:
    model = pysn.Model(mpath)
    print("pysn.Model(%s)" % mpath)
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

    print("\nmodel.parameters.inits:")
    print(model.parameters.inits)

print("\nnm_exe = nonmem.execute.NONMEM7()")
nm_exe = pysn.api_utils.getAPI('nonmem').execute.NONMEM7()
print("bool(nm_exe) = %s" % (bool(nm_exe),))
if nm_exe:
    print('nm_exe.bin = %r (nm_exe.version=%r)' % (nm_exe.bin, nm_exe.version))
print("nm_exe.installed = %r" % (nm_exe.installed,))

# model = pysn.Model(mpath.parent / 'TEST' / 'test.mod')
# model.estimate()
