from __future__ import annotations

from functools import reduce
from itertools import chain
from operator import __and__, is_
from typing import Dict, Iterable, List, Set

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
    if (
        (
            simultaneous
            or set(_mapping.keys()).isdisjoint(
                set().union(*map(lambda e: e.free_symbols, _mapping.values()))
            )
        )
        and all(map(_old_does_not_need_generic_subs, _mapping.keys()))
        and all(map(_new_does_not_need_generic_subs, _mapping.values()))
    ):
        return subs_symbols(expr, _mapping)
    return expr.subs(_mapping, simultaneous=simultaneous)


def _old_does_not_need_generic_subs(expr: sympy.Expr):
    return isinstance(expr, sympy.Symbol)


def _new_does_not_need_generic_subs(expr: sympy.Expr):
    return isinstance(expr, (sympy.Symbol, sympy.Number))


def subs_symbols(expr: sympy.Expr, mapping: Dict[sympy.Symbol, sympy.Expr]):
    stack = [expr]
    output = [[], []]

    while stack:

        e = stack[-1]
        old_args = e.args

        new_args = output[-1]

        n = len(old_args)
        i = len(new_args)

        assert i <= n

        if i == n:
            stack.pop()
            output.pop()
            output[-1].append(_subs_new_args(mapping, e, new_args))
        else:
            # NOTE Push the next argument on the stack
            stack.append(old_args[i])
            output.append([])

    return output[0][0]


def _subs_new_args(
    mapping: Dict[sympy.Symbol, sympy.Expr], expr: sympy.Expr, args: List[sympy.Expr]
):
    assert len(expr.args) == len(args)

    if isinstance(expr, sympy.Symbol):
        return mapping.get(expr, expr)

    if all(map(is_, expr.args, args)):
        return expr

    return expr.func(*args)
