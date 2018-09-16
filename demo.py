#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import importlib
import sys
from pathlib import Path

import pandas as pd


# load 'src/pharmpy' path as module 'pharmpy'
root = Path(__file__).resolve().parent
sys.path.append(str(root / 'src'))
pharmpy = importlib.import_module('pharmpy')
sys.path.pop()

# set pheno path and load
testpath = root / 'tests' / 'testdata' / 'nonmem'
pheno_path = testpath / 'pheno_real.mod'
pheno = pharmpy.Model(pheno_path)

# print pheno representation
print('repr(pheno) = %r' % pheno)

# print THETA records
print("recs = pheno.get_records('THETA')")
recs = pheno.get_records('THETA')
for i, rec in enumerate(recs):
    for j, theta in enumerate(rec.thetas):
        print('recs[%d].thetas[%d] = %s' % (i, j, theta))

# print THETA parse tree
print("\nmodel.get_records('THETA')[0].parser =")
print(recs[0].parser)

# print INPUT data frame
print("\nmodel.input.data_frame =")
pd.set_option('display.max_rows', 10)
print(pheno.input.data_frame)

# print initial values
print("\nmodel.parameters.inits:")
print(pheno.parameters.inits)

# print source resource representation
print('repr(pheno.source) = %r' % pheno.source)

# print NONMEM installations
print("bool(pheno.execute) = %s" % (bool(pheno.execute),))
if pheno.execute:
    print('pheno.execute.bin = %r (pheno.execute.version=%r)' % (pheno.execute.bin,
                                                                 pheno.execute.version))
print("pheno.execute.installed = %r" % (pheno.execute.installed,))

# estimate pheno
# import pdb; pdb.set_trace()  # noqa
# pheno.execute.estimate()
