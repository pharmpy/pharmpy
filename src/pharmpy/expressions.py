from __future__ import annotations

from functools import reduce
from itertools import chain
from operator import __and__, is_
from typing import Dict, Iterable, Set

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


def xreplace_dict(dictlike):
    return {sympify_old(key): sympify_new(value) for key, value in dictlike.items()}


def sympify_old(old):
    # NOTE This mimics sympy's input coercion in subs
    return (
        sympy.Symbol(old)
        if isinstance(old, str)
        else sympy.sympify(old, strict=not isinstance(old, type))
    )


def sympify_new(new):
    # NOTE This mimics sympy's input coercion in subs
    return sympy.sympify(new, strict=not isinstance(new, (str, type)))


def subs(expr: sympy.Expr, mapping: Dict[sympy.Expr, sympy.Expr], simultaneous: bool = False):
    _mapping = xreplace_dict(mapping)
    if simultaneous and all(
        map(_does_not_need_generic_subs, chain(_mapping.keys(), _mapping.values()))
    ):
        return subs_symbols(expr, _mapping)
    return expr.subs(_mapping, simultaneous=simultaneous)


def _does_not_need_generic_subs(expr: sympy.Expr):
    return isinstance(expr, (sympy.Symbol, sympy.Number))


def subs_symbols(expr: sympy.Expr, mapping: Dict[sympy.Symbol, sympy.Expr]):
    if isinstance(expr, sympy.Symbol):
        return mapping.get(expr, expr)

    if not expr.args:
        return expr

    new_args = tuple(subs_symbols(arg, mapping) for arg in expr.args)

    if all(map(is_, expr.args, new_args)):
        return expr

    return expr.func(*new_args)
