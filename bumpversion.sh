#!/usr/bin/env bash

if [ ! -e '.git' ] ; then
    >&2 printf 'Not at the root of the git repository!\n'
    >&2 printf 'Not doing anything.\n'
    >&2 printf 'Exiting.\n'
    exit 1
fi

if [ -n "$(git status --porcelain=v1 2>/dev/null)" ] ; then
    >&2 printf 'git status is dirty!\n'
    >&2 printf 'Not doing anything.\n'
    >&2 printf 'Exiting.\n'
    exit 1
fi

year=`date +"%Y"`
prevyear=$(($year-1))
sed -i "s/2018-$prevyear/2018-$year/" docs/conf.py
sed -i "s/2018-$prevyear/2018-$year/" docs/license.rst
if [ -n "$(git status --porcelain=v1 2>/dev/null)" ] ; then
    git add docs/conf.py docs/license.rst
    git commit -m "Updating copyright years"
fi

bumpversion $1
