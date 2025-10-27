from __future__ import annotations

from functools import reduce
from operator import __and__
from typing import Iterable

from pharmpy.deps import sympy

from .leaves import free_images_and_symbols
from .subs import subs


def assume_all(predicate: sympy.assumptions.Predicate, expressions: Iterable[sympy.Expr]):
    tautology = sympy.Q.is_true(True)
    return reduce(__and__, map(predicate, expressions), tautology)


def posify(expr: sympy.Expr):
    if expr.is_positive is not None:
        return expr

    if isinstance(expr, sympy.Symbol):
        return sympy.Symbol(expr.name, positive=True, **expr.assumptions0)

    if isinstance(expr.func, sympy.core.function.UndefinedFunction):
        return sympy.core.function.UndefinedFunction(expr.func.name, positive=True)(*expr.args)

    return subs(expr, {x: posify(x) for x in free_images_and_symbols(expr)}, simultaneous=True)
