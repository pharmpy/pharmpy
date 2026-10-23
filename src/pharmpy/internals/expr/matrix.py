# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from pharmpy.deps import sympy


def is_zero_matrix(A: sympy.Matrix) -> bool:
    for e in A:
        assert isinstance(e, sympy.Expr)
        if not e.is_zero:
            return False
    return True
