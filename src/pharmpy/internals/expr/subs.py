from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List, Mapping

from pharmpy.deps import sympy

from .tree import replace_root_children


def subs(expr: sympy.Expr, mapping: Mapping[sympy.Expr, sympy.Expr], simultaneous: bool = False):
    _mapping = xreplace_dict(mapping)
    if (simultaneous or _mapping_is_not_recursive(_mapping)) and all(
        map(_old_does_not_need_generic_subs, _mapping.keys())
    ):
        if sympy.exp in _mapping:
            new_base = _mapping[sympy.exp]
            _mapping[sympy.exp] = lambda x: new_base**x
            _mapping[sympy.Pow] = lambda a, x: new_base**x if a is sympy.S.Exp1 else a**x
            return _subs_atoms_simultaneously(_subs_atom_or_func(_mapping), expr)
        return _subs_atoms_simultaneously(_subs_atom(_mapping), expr)
    return expr.subs(_mapping, simultaneous=simultaneous)


def xreplace_dict(dictlike) -> Dict[sympy.Expr, sympy.Expr]:
    return {_sympify_old(key): _sympify_new(value) for key, value in dictlike.items()}


def _sympify_old(old) -> sympy.Expr:
    # NOTE This mimics sympy's input coercion in subs
    return (
        sympy.Symbol(old)
        if isinstance(old, str)
        else sympy.sympify(old, strict=not isinstance(old, type))
    )


@lru_cache(maxsize=256)
def _sympify_new(new) -> sympy.Expr:
    # NOTE This mimics sympy's input coercion in subs
    return sympy.sympify(new, strict=not isinstance(new, (str, type)))


def _mapping_is_not_recursive(mapping: Dict[sympy.Expr, sympy.Expr]):
    return set(mapping.keys()).isdisjoint(
        set().union(*map(lambda e: e.free_symbols, mapping.values()))
    )


def _old_does_not_need_generic_subs(expr: sympy.Expr):
    return isinstance(expr, sympy.Symbol) or expr is sympy.exp


def _subs_atoms_simultaneously(
    subs_new_args: Callable[[sympy.Expr, List[sympy.Expr]], sympy.Expr], expr: sympy.Expr
):
    stack = [expr]
    output = [[], []]

    while stack:
        e = stack[-1]
        old_args = e.args

        new_args = output[-1]

        n = len(old_args)
        i = len(new_args)

        if i == n:
            stack.pop()
            output.pop()
            output[-1].append(subs_new_args(e, new_args))
        else:
            # NOTE Push the next argument on the stack
            stack.append(old_args[i])
            output.append([])

    return output[0][0]


def _subs_atom(mapping: Dict[sympy.Expr, sympy.Expr]):
    def _subs(expr: sympy.Expr, args: List[sympy.Expr]):
        return replace_root_children(expr, args) if args else mapping.get(expr, expr)

    return _subs


def _subs_atom_or_func(mapping: Dict[sympy.Expr, sympy.Expr]):
    def _subs(expr: sympy.Expr, args: List[sympy.Expr]):
        if not args:
            return mapping.get(expr, expr)

        fn = expr.func
        new_fn = mapping.get(fn, fn)
        return replace_root_children(expr, args) if fn is new_fn else new_fn(*args)

    return _subs
