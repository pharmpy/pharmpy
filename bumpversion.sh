#!/bin/bash
bumpversion $1
year=`date +"%Y"`
prevyear=$(($year-1))
sed -i "s/2018-$prevyear/2018-$year/" docs/conf.py
sed -i "s/2018-$prevyear/2018-$year/" docs/license.rst
git commit -a --amend --no-edit
