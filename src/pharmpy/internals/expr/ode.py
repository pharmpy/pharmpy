from __future__ import annotations

from pharmpy.deps import sympy

from .leaves import free_images


def canonical_ode_rhs(expr: sympy.Expr):
    fi = free_images(expr)
    return sympy.collect(_expand_rates(expr, fi), sorted(fi, key=str))


def _expand_rates(expr: sympy.Expr, free_images: set[sympy.Expr]):
    if isinstance(expr, sympy.Add):

        def func(x):
            return _expand_rates(x, free_images)

        return sympy.expand(
            sympy.Add(*map(func, expr.args)),
            deep=False,
        )
    if (
        isinstance(expr, sympy.Mul)
        and len(expr.args) == 2
        and not free_images.isdisjoint(expr.args)
    ):
        return sympy.expand(expr, deep=False)
    return expr
