#!/bin/bash

# This script will create a temporary directory and run tools there, then results.json will overwrite
# each example tool results respectively

rm -rf /tmp/tool_results

cd ..
mkdir /tmp/tool_results

cp tests/testdata/nonmem/models/mox2.mod /tmp/tool_results/
cp tests/testdata/nonmem/models/mox2.lst /tmp/tool_results/
cp tests/testdata/nonmem/models/mox2.ext /tmp/tool_results/
cp tests/testdata/nonmem/models/mox2.phi /tmp/tool_results/
cp tests/testdata/nonmem/models/mox_simulated_normal.csv /tmp/tool_results/

tox -e run -- pharmpy run modelsearch /tmp/tool_results/mox2.mod 'ABSORPTION(ZO);PERIPHERALS(1)' 'exhaustive_stepwise' --path /tmp/tool_results/modelsearch/
cp /tmp/tool_results/modelsearch/results.json tests/testdata/results/modelsearch_results.json

exit
