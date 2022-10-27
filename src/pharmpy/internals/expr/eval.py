from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Set

from pharmpy.deps import numpy as np
from pharmpy.deps import sympy

from .subs import subs


def eval_expr(
    expr: sympy.Expr,
    datasize: int,
    datamap: Mapping[sympy.Symbol, np.ndarray],
) -> np.ndarray:
    # NOTE We avoid querying for free_symbols if we know none are expected
    fs = _free_symbols(expr) if datamap else set()

    if fs:
        ordered_symbols, fn = _lambdify_canonical(expr)
        data = [datamap[rv] for rv in ordered_symbols]
        return fn(*data)

    return np.full(datasize, float(expr.evalf()))


@lru_cache(maxsize=256)
def _free_symbols(expr: sympy.Expr) -> Set[sympy.Symbol]:
    return expr.free_symbols


@lru_cache(maxsize=256)
def _lambdify_canonical(expr: sympy.Expr):
    fs = _free_symbols(expr)
    ordered_symbols = sorted(fs, key=str)
    # NOTE Substitution allows to use cse. Otherwise weird things happen with
    # symbols that look like function eval (e.g. ETA(1), THETA(3), OMEGA(1,1)).
    ordered_substitutes = [sympy.Symbol(f'__tmp{i}') for i in range(len(ordered_symbols))]
    substituted_expr = subs(
        expr,
        dict(zip(ordered_symbols, ordered_substitutes)),
        simultaneous=True,
    )
    fn = sympy.lambdify(ordered_substitutes, substituted_expr, modules='numpy', cse=True)
    return ordered_symbols, fn
