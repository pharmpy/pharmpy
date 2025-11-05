from __future__ import annotations

from functools import reduce
from operator import __and__
from typing import Iterable

from pharmpy.deps import sympy

from .subs import subs


def assume_all(predicate: sympy.assumptions.Predicate, expressions: Iterable[sympy.Expr]):
    tautology = sympy.Q.is_true(True)
    return reduce(__and__, map(predicate, expressions), tautology)


def _free_image_or_symbol_with_assumptions(expr: sympy.Basic, assumptions: dict[str, bool]):
    if isinstance(expr, sympy.Symbol):
        return sympy.Symbol(expr.name, **assumptions, **expr.assumptions0)

    if isinstance(expr.func, sympy.core.function.UndefinedFunction):
        return sympy.core.function.UndefinedFunction(
            expr.func.name, **assumptions, **expr.assumptions0
        )(*expr.args)

    raise NotImplementedError(f'{expr}')


def _free_images_and_symbols_assumptions_mapping(assumptions: sympy.logic.boolalg.Boolean):
    if isinstance(assumptions, sympy.And):
        for term in assumptions.args:
            yield from _free_images_and_symbols_assumptions_mapping(term)
        return

    if isinstance(assumptions, sympy.AppliedPredicate):
        predicate, *args = assumptions.args
        if predicate == sympy.Q.positive:
            assert len(args) == 1
            expr = args[0]
            yield [expr, _free_image_or_symbol_with_assumptions(expr, {'positive': True})]
            return

        if predicate == sympy.Q.is_true:
            assert len(args) == 1
            assert args[0]
            return

    raise NotImplementedError(f'{assumptions}')


def with_free_images_and_symbols_assumptions(
    expr: sympy.Expr, assumptions: sympy.logic.boolalg.Boolean
):
    mapping = dict(_free_images_and_symbols_assumptions_mapping(assumptions))
    return subs(expr, mapping, simultaneous=True)
