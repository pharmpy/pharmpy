# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import annotations

from pharmpy.deps import sympy


def parse(expr: int | float | str | sympy.Expr) -> sympy.Expr:
    ns = {'Q': sympy.Symbol('Q'), 'LT': sympy.Symbol('LT'), 'N': sympy.Symbol('N')}
    expr = sympy.sympify(expr, locals=ns)  # pyright: ignore [reportCallIssue]
    assert isinstance(expr, sympy.Expr)
    return expr
