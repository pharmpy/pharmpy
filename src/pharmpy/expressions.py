from __future__ import annotations

from functools import lru_cache, reduce
from itertools import chain
from operator import __and__, is_
from typing import Callable, Dict, Iterable, List, Set

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


def subs(expr: sympy.Expr, mapping: Dict[sympy.Expr, sympy.Expr], simultaneous: bool = False):
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


def _build(expr: sympy.Expr, args: List[sympy.Expr]):
    return expr if all(map(is_, expr.args, args)) else expr.func(*args)


def _subs_atom(mapping: Dict[sympy.Expr, sympy.Expr]):
    def _subs(expr: sympy.Expr, args: List[sympy.Expr]):
        return _build(expr, args) if args else mapping.get(expr, expr)

    return _subs


def _subs_atom_or_func(mapping: Dict[sympy.Expr, sympy.Expr]):
    def _subs(expr: sympy.Expr, args: List[sympy.Expr]):
        if not args:
            return mapping.get(expr, expr)

        fn = expr.func
        new_fn = mapping.get(fn, fn)
        return _build(expr, args) if fn is new_fn else new_fn(*args)

    return _subs


def _neutral(fn: sympy.Function) -> sympy.Integer:
    if fn is sympy.Add:
        return sympy.Integer(0)
    if fn is sympy.Mul:
        return sympy.Integer(1)
    if fn is sympy.Pow:
        return sympy.Integer(1)

    raise ValueError(f'{type(fn)}: {repr(fn)}')


def prune(predicate: Callable[[sympy.Expr], bool], expr: sympy.Expr):
    """Create a new expression by removing subexpressions nodes from an input
    expression.

    Parameters
    ----------
    predicate : (sympy.Expr) -> bool
        A function that takes a subexpression as input and returns a bool
        indicating whether this subexpression should be pruned.
    expr : sympy.Expr
        The input expression.

    Returns
    -------
    sympy.Expr
        The pruned expression.

    Examples
    --------
    >>> import sympy
    >>> from pharmpy.expressions import prune
    >>> prune(lambda expr: expr.func is sympy.exp, sympy.exp(2))
    0
    >>> prune(lambda expr: expr.func is sympy.exp, 3*sympy.exp(2))
    3

    See also
    --------
    subs

    """
    if predicate(expr):
        return _neutral(sympy.Add)

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
            output[-1].append(_build(e, new_args))
        else:
            old_arg = old_args[i]
            if predicate(old_arg):
                # NOTE Replace node without recursing
                new_args.append(_neutral(e.func))
            else:
                # NOTE Push the next argument on the stack
                stack.append(old_arg)
                output.append([])

    return output[0][0]
