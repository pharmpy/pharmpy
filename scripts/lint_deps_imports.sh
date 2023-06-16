#!/usr/bin/env sh

matches () {
    grep -E \
        -e '^import altair($|\.| )' -e '^from altair( |\.)' \
        -e '^import networkx($|\.| )' -e '^from networkx( |\.)' \
        -e '^import numpy($|\.| )' -e '^from numpy( |\.)' \
        -e '^import pandas($|\.| )' -e '^from pandas( |\.)' \
        -e '^import rich($|\.| )' -e '^from rich( |\.)' \
        -e '^import scipy($|\.| )' -e '^from scipy( |\.)' \
        -e '^import symengine($|\.| )' -e '^from symengine( |\.)' \
        -e '^import sympy($|\.| )' -e '^from sympy( |\.)' \
        -r src/pharmpy \
        --exclude-dir deps \
        --include="*.py" \
        "$@"
}

nmatches="$(matches -o | wc -l)"

if [ "$nmatches" -gt 0 ] ; then
    printf 'The following %d lines use non-lazy imports:\n\n' "$nmatches"
    matches -n --color=auto
    exit 1
fi

exit 0
