from __future__ import annotations

from functools import reduce
from itertools import chain
from operator import __and__
from typing import Iterable, Set

from pharmpy.deps import sympy


def sympify(expr):
    ns = {'Q': sympy.Symbol('Q')}
    return sympy.sympify(expr, locals=ns)


def canonical_ode_rhs(expr: sympy.Expr):
    fi = free_images(expr)
    return sympy.collect(_expand_rates(expr, fi), sorted(fi, key=str))


def _expand_rates(expr: sympy.Expr, free_images: Set[sympy.Expr]):
    if isinstance(expr, sympy.Add):
        return sympy.expand(
            sympy.Add(*map(lambda x: _expand_rates(x, free_images), expr.args)), deep=False
        )
    if (
        isinstance(expr, sympy.Mul)
        and len(expr.args) == 2
        and not free_images.isdisjoint(expr.args)
    ):
        return sympy.expand(expr, deep=False)
    return expr


def free_images_and_symbols(expr: sympy.Expr) -> Set[sympy.Expr]:
    return expr.free_symbols | free_images(expr)


def free_images(expr: sympy.Expr) -> Set[sympy.Expr]:
    return set(_free_images_iter(expr))


def _free_images_iter(expr: sympy.Expr) -> Iterable[sympy.Expr]:
    if isinstance(expr.func, sympy.core.function.UndefinedFunction):
        yield expr
        return

    yield from chain.from_iterable(
        map(
            _free_images_iter,
            expr.args,
        )
    )


def assume_all(predicate: sympy.assumptions.Predicate, expressions: Iterable[sympy.Expr]):
    tautology = sympy.Q.is_true(True)
    return reduce(__and__, map(predicate, expressions), tautology)
