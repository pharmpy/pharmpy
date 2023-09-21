from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import sympy
else:
    from pharmpy.deps import sympy


def parse(expr: Union[int, float, str, sympy.Basic]) -> sympy.Basic:
    ns = {'Q': sympy.Symbol('Q'), 'LT': sympy.Symbol('LT')}
    expr = sympy.sympify(expr, locals=ns)
    assert isinstance(expr, sympy.Basic)
    return expr
