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


UPDATED_FILES=(docs/{conf.py,license.rst})
>&2 printf 'Checking the following files for copyright years update\n'
>&2 printf '  - %s\n' "${UPDATED_FILES[@]}"
year="$(date +"%Y")"
prevyear=$((year-1))
sed -i "s/2018-${prevyear}/2018-${year}/" "${UPDATED_FILES[@]}"
if [ -n "$(git status --porcelain=v1 2>/dev/null)" ] ; then
    git add "${UPDATED_FILES[@]}"
    git commit -m "Updating copyright years"
fi

>&2 printf 'Bumping version with bumpversion'
bumpversion "$@"
