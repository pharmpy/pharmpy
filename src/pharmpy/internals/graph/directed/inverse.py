# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later


def inverse[T, U](g: dict[T, set[U]]) -> dict[U, set[T]]:
    h = {}

    for left, deps in g.items():
        for right in deps:
            if right in h:
                h[right].add(left)
            else:
                h[right] = {left}

    return h
