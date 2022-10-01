#!/bin/bash

# This script will create a temporary directory and run tools there, then results.json will overwrite
# each example tool results respectively

while getopts t: opt; do
    case "$opt" in
        t ) TOOL=${OPTARG};;
        * ) exit 1;;
    esac
done


TESTPATH=/tmp/tool_results
TESTDATA=tests/testdata/nonmem
DEST=tests/testdata/results
if [ -z "$TOOL" ]; then
  TOOL='all'
fi

echo "Running tools: $TOOL"

rm -rf $TESTPATH

cd ..
mkdir $TESTPATH

python3 -m venv $TESTPATH/pharmpy_venv
activate () {
    . $TESTPATH/pharmpy_venv/bin/activate
}
activate
pip install .
pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtime

cp_results () {
  # This converts MacOS-style temp paths to Linux-style ones
  sed -E 's:"/private/tmp/:"/tmp/:g' "$1" > "$2"
}

cp $TESTDATA/pheno.mod $TESTPATH
cp $TESTDATA/pheno.dta $TESTPATH
cp $TESTDATA/pheno.lst $TESTPATH
cp $TESTDATA/pheno.ext $TESTPATH
cp $TESTDATA/pheno.phi $TESTPATH

if [ "$TOOL" == 'allometry' ] || [ "$TOOL" == 'all' ]; then
  pharmpy run allometry $TESTPATH/pheno.mod --allometric_variable=WGT --path $TESTPATH/allometry/
  cp_results $TESTPATH/allometry/results.json $DEST/allometry_results.json
fi

cp $TESTDATA/models/mox2.mod $TESTPATH
cp $TESTDATA/models/mox2.lst $TESTPATH
cp $TESTDATA/models/mox2.ext $TESTPATH
cp $TESTDATA/models/mox2.phi $TESTPATH
cp $TESTDATA/models/mox_simulated_normal.csv $TESTPATH


if [ "$TOOL" == 'modelsearch' ] || [ "$TOOL" == 'all' ]; then
  pharmpy run modelsearch $TESTPATH/mox2.mod 'PERIPHERALS(1);LAGTIME()' 'reduced_stepwise' --path $TESTPATH/modelsearch/
  cp_results $TESTPATH/modelsearch/results.json $DEST/modelsearch_results.json
  cp $TESTPATH/modelsearch/metadata.json $DEST/metadata.json
  cp -r $TESTPATH/modelsearch $DEST/tool_databases
fi

if [ "$TOOL" == 'iivsearch' ] || [ "$TOOL" == 'all' ]; then
  pharmpy run iivsearch $TESTPATH/mox2.mod 'brute_force' --path $TESTPATH/iivsearch/
  cp_results $TESTPATH/iivsearch/results.json $DEST/iivsearch_results.json
fi

if [ "$TOOL" == 'iovsearch' ] || [ "$TOOL" == 'all' ]; then
  pharmpy run iovsearch $TESTPATH/mox2.mod --column 'VISI' --path $TESTPATH/iovsearch/
  cp_results $TESTPATH/iovsearch/results.json $DEST/iovsearch_results.json
fi

if [ "$TOOL" == 'covsearch' ] || [ "$TOOL" == 'all' ]; then
  pharmpy run covsearch $TESTPATH/mox2.mod \
        --effects 'COVARIATE([CL, MAT, VC], [AGE, WT], EXP);COVARIATE([CL, MAT, VC], [SEX], CAT)' \
        --path $TESTPATH/covsearch/
  cp_results $TESTPATH/covsearch/results.json > $DEST/covsearch_results.json
fi

if [ "$TOOL" == 'resmod' ] || [ "$TOOL" == 'all' ]; then
  cp $TESTDATA/resmod/mox3.* $TESTPATH
  cp $TESTDATA/resmod/moxo_simulated_resmod.csv $TESTPATH
  cp $TESTDATA/resmod/mytab $TESTPATH
  pharmpy run resmod $TESTPATH/mox3.mod --path $TESTPATH/resmod/
  cp_results $TESTPATH/resmod/results.json $DEST/resmod_results.json
fi
