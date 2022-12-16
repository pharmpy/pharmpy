from __future__ import annotations

from typing import Union

from pharmpy.deps import sympy


def parse(expr: Union[int, float, str, sympy.Expr]) -> sympy.Expr:
    ns = {'Q': sympy.Symbol('Q'), 'LT': sympy.Symbol('LT')}
    return sympy.sympify(expr, locals=ns)
