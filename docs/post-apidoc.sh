#!/usr/bin/env bash

if [[ -d $1 ]]; then
    DIR="$1"
else
    echo "'$1' is not valid target"
    exit 1
fi

shopt -s nullglob dotglob
files=( "$DIR"/*pharmpy*.rst )

for file in ${files[@]}; do
    if grep -q ".. inheritance-diagram::" "$file"; then
        echo "Found inheritance-diagram in $file."
    elif head -1 "$file" | grep -q -v -E "package$"; then
        automodule=$(grep -E '^\.\. automodule::' "$file")
        if [ -z "$automodule" ]; then
            continue
        fi
        echo "Adding inheritance diagram to $file"
        diagram="$(echo "$automodule" | sed -E "s%.+::(.*)$%.. inheritance-diagram::\\1\\n    :parts: 4%g")"
        printf "\nInheritance Diagram\n-------------------\n\n${diagram}\n" >> "$file"
    fi
done

