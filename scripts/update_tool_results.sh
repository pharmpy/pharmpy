#!/bin/bash

# This script will create a temporary directory and run tools there, then results.json will overwrite
# each example tool results respectively

TESTPATH=/tmp/tool_results
TESTDATA=tests/testdata/nonmem
DEST=tests/testdata/results

rm -rf $TESTPATH

cd ..
mkdir $TESTPATH


cp $TESTDATA/pheno.mod $TESTPATH
cp $TESTDATA/pheno.dta $TESTPATH
cp $TESTDATA/pheno.lst $TESTPATH
cp $TESTDATA/pheno.ext $TESTPATH
cp $TESTDATA/pheno.phi $TESTPATH

tox -e run -- pharmpy run allometry $TESTPATH/pheno.mod --allometric_variable=WGT --path $TESTPATH/allometry/
cp $TESTPATH/allometry/results.json $DEST/allometry_results.json

cp $TESTDATA/models/mox2.mod $TESTPATH
cp $TESTDATA/models/mox2.lst $TESTPATH
cp $TESTDATA/models/mox2.ext $TESTPATH
cp $TESTDATA/models/mox2.phi $TESTPATH
cp $TESTDATA/models/mox_simulated_normal.csv $TESTPATH

tox -e run -- pharmpy run modelsearch $TESTPATH/mox2.mod 'ABSORPTION(ZO);PERIPHERALS(1)' 'exhaustive_stepwise' --path $TESTPATH/modelsearch/
cp $TESTPATH/modelsearch/results.json $DEST/modelsearch_results.json

tox -e run -- pharmpy run iivsearch $TESTPATH/mox2.mod 'brute_force' --path $TESTPATH/iivsearch/
cp $TESTPATH/iivsearch/results.json $DEST/iivsearch_results.json

tox -e run -- pharmpy run iovsearch $TESTPATH/mox2.mod --column 'VISI' --path $TESTPATH/iovsearch/
cp $TESTPATH/iovsearch/results.json $DEST/iovsearch_results.json

cp $TESTDATA/resmod/mox3.* $TESTPATH
cp $TESTDATA/resmod/moxo_simulated_resmod.csv $TESTPATH
cp $TESTDATA/resmod/mytab $TESTPATH
tox -e run -- pharmpy run resmod $TESTPATH/mox3.mod --path $TESTPATH/resmod/
cp $TESTPATH/resmod/results.json $DEST/resmod_results.json
