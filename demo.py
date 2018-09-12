#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import importlib
import sys
from pathlib import Path

import pandas as pd


# load 'src/pysn' path as module 'pysn'
root = Path(__file__).resolve().parent
sys.path.append(str(root / 'src'))
pysn = importlib.import_module('pysn')
sys.path.pop()

# set pheno path and load
testpath = root / 'tests' / 'testdata' / 'nonmem'
pheno_path = testpath / 'pheno_real.mod'
print("pysn.Model(%s)" % pheno_path)
pheno = pysn.Model(pheno_path)
print('='*80)

# print pheno content
print('str(pheno) =')
lines = str(pheno).splitlines()
for i, line in enumerate(lines):
    print('%3d: %r' % (i, repr(line)))

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

# print NONMEM installations
print("\nnm_exe = nonmem.execute.NONMEM7()")
nm_exe = pysn.api_utils.getAPI('nonmem').execute.NONMEM7()
print("bool(nm_exe) = %s" % (bool(nm_exe),))
if nm_exe:
    print('nm_exe.bin = %r (nm_exe.version=%r)' % (nm_exe.bin, nm_exe.version))
print("nm_exe.installed = %r" % (nm_exe.installed,))

# estimate pheno
# import pdb; pdb.set_trace()  # noqa
# pheno.estimate()
