from __future__ import annotations

from operator import is_
from typing import Callable, List

from pharmpy.deps import sympy


def replace_root_children(expr: sympy.Expr, args: List[sympy.Expr]):
    # NOTE This creates a new tree by replacing the children of the root node.
    # If the children have not changed it returns the original tree which
    # allows certain downstream optimizations.
    return expr if all(map(is_, expr.args, args)) else expr.func(*args)


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
    >>> from pharmpy.internals.expr.tree import prune
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
            output[-1].append(replace_root_children(e, new_args))
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


def _neutral(fn: sympy.Function) -> sympy.Integer:
    if fn is sympy.Add:
        return sympy.Integer(0)
    if fn is sympy.Mul:
        return sympy.Integer(1)
    if fn is sympy.Pow:
        return sympy.Integer(1)

    raise ValueError(f'{type(fn)}: {repr(fn)}')
