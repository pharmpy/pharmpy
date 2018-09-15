#!/usr/bin/env bash

function INFO() {
    echo "$(tput bold; tput setaf 7)MESSAGE$(tput sgr0) $@"
}
function EXEC() {
    echo "$(tput bold; tput setaf 3)EXECUTE$(tput sgr0) $@"
}
function ERR() {
    echo "$(tput bold; tput setaf 1)ERROR$(tput sgr0) $@"
}

function cmd() {
    local note="$(command -v "$1")"
    if [ ! -n "$note" ]; then
        note="$1"
        shift
    fi
    EXEC "$note"
    echo -n "  "
    set -x; "$@"; { set +x; } 2>/dev/null
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
if [ "$DIR" != "$(pwd)" ]; then
    INFO "Target dir: $DIR"
    cmd "Changing dir" cd "$DIR"
fi
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    ERR "Refusing! Target not Git repo: $DIR"
    exit 1
elif [ "$(git cat-file -t 0db96cd4db3aa193715322b61e120087a8b2e922 2>/dev/null)" != "commit" ]; then
    ERR "Refusing! Target not the PharmPy repo: $DIR"
    exit 1
fi

cmd "Removing autogen docs (docs/reference/)" rm -rf "$DIR"/docs/reference/
cmd "Removing build dir (dist/)"              rm -rf "$DIR"/dist/
cmd "Removing coverage dir (htmlcov/)"        rm -rf "$DIR"/htmlcov/
cmd "Removing egg info dirs (src/*.egg-info)" rm -rf "$DIR"/src/*.egg-info/
cmd "Removing pytest cache (.pytest_cache/)"  rm -rf "$DIR"/.pytest_cache/
cmd "Removing Tox cache (.tox/)"              rm -rf "$DIR"/.tox/
