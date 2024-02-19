from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import sympy
else:
    from pharmpy.deps import sympy


def free_images(expr: sympy.Expr) -> set[sympy.Expr]:
    return set(_free_images_iter(expr))


def free_images_and_symbols(expr: sympy.Expr) -> set[sympy.Expr]:
    return expr.free_symbols | free_images(expr)  # pyright: ignore [reportReturnType]


def _free_images_iter(expr: sympy.Expr) -> Iterable[sympy.Expr]:
    if isinstance(expr.func, sympy.core.function.UndefinedFunction):
        yield expr
        return

    yield from chain.from_iterable(
        map(
            _free_images_iter,
            expr.args,  # pyright: ignore [reportArgumentType]
        )
    )
