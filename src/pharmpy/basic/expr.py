from __future__ import annotations

from collections.abc import Mapping
from typing import Union

from pharmpy.deps import symengine, sympy
from pharmpy.deps.sympy_printing import pretty
from pharmpy.internals.expr.assumptions import assume_all
from pharmpy.internals.expr.leaves import free_images_and_symbols


class ExprPrinter(pretty.PrettyPrinter):
    def __init__(self):
        super().__init__(settings={'wrap_line': False, 'use_unicode': True})

    def _print_Equality(self, e):
        # symengine can turn some Equations around
        if isinstance(e.rhs, sympy.Symbol):
            lhs = e.rhs
            rhs = e.lhs
        else:
            lhs = e.lhs
            rhs = e.rhs
        return super()._print_Relational(sympy.Eq(lhs, rhs))


class Expr:
    """A real valued symbolic expression with real symbols"""

    def __init__(self, source: TExpr):
        if isinstance(source, Expr):
            self._expr = source._expr
        else:
            try:
                self._expr = symengine.sympify(source)
            except symengine.SympifyError:
                raise TypeError(
                    f'Cannot convert input to Expr: expression `{source}` of type `{type(source)}`'
                )

    @property
    def name(self) -> str:
        if isinstance(self._expr, symengine.Symbol):
            return self._expr.name
        elif isinstance(self._expr, symengine.Function):
            try:
                return self._expr.get_name()
            except AttributeError:
                return str(self._expr).partition("(")[0]
        else:
            raise ValueError("Expression has no name")

    @property
    def args(self) -> tuple[Union[Expr, tuple[Union[Expr, BooleanExpr], ...]], ...]:
        if isinstance(self._expr, symengine.Piecewise):
            # Due to https://github.com/symengine/symengine.py/issues/469
            x = self._expr.args
            args = tuple((Expr(x[i]), BooleanExpr(x[i + 1])) for i in range(0, len(x), 2))
        else:
            args = tuple(Expr(a) for a in self._expr.args)
        return args

    @property
    def piecewise_args(self) -> tuple[tuple[Expr, BooleanExpr], ...]:
        if isinstance(self._expr, symengine.Piecewise):
            x = self._expr.args
            args = tuple((Expr(x[i]), BooleanExpr(x[i + 1])) for i in range(0, len(x), 2))
        else:
            raise ValueError("Expression is not a piecewise")
        return args

    def make_args(self, expr):
        return sympy.sympify(self._expr).make_args(sympy.sympify(expr))

    @property
    def free_symbols(self) -> set[Expr]:
        symbs = {Expr(a) for a in self._expr.free_symbols}
        return symbs

    def subs(self, d: Mapping[TExpr, TExpr]) -> Expr:
        return Expr(self._expr.subs(d))

    def as_numer_denom(self) -> tuple[Expr, Expr]:
        numer, denom = sympy.sympify(self._expr).as_numer_denom()
        return Expr(numer), Expr(denom)

    def simplify(self) -> Expr:
        return Expr(sympy.sympify(self._expr).simplify())

    def expand(self) -> Expr:
        # NOTE: the expression exp(x+y) will be expanded to
        #  exp(x)+exp(y) in sympy but kept as exp(x+y) in symengine
        return Expr(self._expr.expand())

    def __add__(self, other) -> Expr:
        return Expr(self._expr + other)

    def __radd__(self, other) -> Expr:
        return Expr(other + self._expr)

    def __sub__(self, other) -> Expr:
        return Expr(self._expr - other)

    def __rsub__(self, other) -> Expr:
        return Expr(other - self._expr)

    def __mul__(self, other) -> Expr:
        return Expr(self._expr * other)

    def __rmul__(self, other) -> Expr:
        return Expr(other * self._expr)

    def __truediv__(self, other) -> Expr:
        return Expr(self._expr / other)

    def __rtruediv__(self, other) -> Expr:
        return Expr(other / self._expr)

    def __pow__(self, other) -> Expr:
        return Expr(self._expr**other)

    def __rpow__(self, other) -> Expr:
        return Expr(other**self._expr)

    def __neg__(self) -> Expr:
        return Expr(-self._expr)

    def __pos__(self) -> Expr:
        return Expr(self._expr)

    def __abs__(self) -> Expr:
        return Expr(abs(self._expr))

    def __eq__(self, other) -> bool:
        return self._expr == other

    def __hash__(self):
        return hash(self._expr)

    def __float__(self) -> float:
        return float(self._expr)

    def __int__(self) -> int:
        return int(self._expr)

    def __bool__(self) -> bool:
        return self._expr != 0

    def __repr__(self) -> str:
        return repr(sympy.sympify(self._expr))

    def serialize(self) -> str:
        return sympy.srepr(sympy.sympify(self._expr))

    @classmethod
    def deserialize(cls, s: str) -> Expr:
        return cls(sympy.parse_expr(s))

    def unicode(self) -> str:
        s = sympy.pretty(sympy.sympify(self._expr), wrap_line=False, use_unicode=True)
        return s

    def latex(self) -> str:
        expr = sympy.sympify(self._expr)
        s = sympy.latex(expr, mul_symbol='dot')
        return s

    def _sympy_(self) -> sympy.core.expr.Expr:
        return sympy.sympify(self._expr)

    def _symengine_(self) -> symengine.Expr:
        return self._expr

    def exp(self) -> Expr:
        return Expr(symengine.exp(self._expr))

    def log(self) -> Expr:
        return Expr(symengine.log(self._expr))

    def sqrt(self) -> Expr:
        return Expr(symengine.sqrt(self._expr))

    def sign(self) -> Expr:
        return Expr(symengine.sign(self._expr))

    def diff(self, x) -> Expr:
        return Expr(self._expr.diff(x._expr))

    def is_symbol(self) -> bool:
        # NOTE: The concept of a symbol is wider than that of sympy and symengine
        return (
            self._expr.is_Symbol
            or self._expr.is_Derivative
            or isinstance(self._expr, symengine.FunctionSymbol)
        )

    def is_integer(self) -> bool:
        return isinstance(self._expr, symengine.Integer)

    def is_number(self) -> bool:
        return isinstance(self._expr, symengine.Number)

    def is_mul(self) -> bool:
        return hasattr(self._expr, 'func') and self._expr.func == symengine.Mul

    def is_add(self) -> bool:
        return hasattr(self._expr, 'func') and self._expr.func == symengine.Add

    def is_exp(self) -> bool:
        return self.is_pow() and self._expr.args[0] == symengine.E

    def is_pow(self) -> bool:
        return isinstance(self._expr, symengine.Pow)

    def is_function(self) -> bool:
        return self._expr.is_Function

    def is_derivative(self) -> bool:
        return hasattr(self._expr, 'func') and self._expr.func == symengine.Derivative

    def is_piecewise(self) -> bool:
        return isinstance(self._expr, symengine.Piecewise)

    def is_nonnegative(self) -> bool | None:
        return sympy.ask(sympy.Q.nonnegative(self._expr))

    def is_real(self) -> bool | None:
        sympy_expr = self._sympy_()
        return sympy.ask(
            sympy.Q.real(sympy_expr), assume_all(sympy.Q.real, free_images_and_symbols(sympy_expr))
        )

    def piecewise_fold(self) -> Expr:
        if isinstance(self._expr, symengine.Piecewise):
            expr = sympy.sympify(self._expr)
            expr = sympy.piecewise_fold(expr)
            return Expr(expr)
        else:
            return self

    @classmethod
    def symbol(cls, name: str) -> Expr:
        symb = symengine.Symbol(name)
        return cls(symb)

    @classmethod
    def dummy(cls, name: str) -> Expr:
        symb = symengine.Dummy(name)
        return cls(symb)

    @classmethod
    def integer(cls, value: int) -> Expr:
        n = symengine.Integer(value)
        return cls(n)

    @classmethod
    def float(cls, value: float) -> Expr:
        x = symengine.RealDouble(value)
        return cls(x)

    @classmethod
    def derivative(cls, f, *x) -> Expr:
        dfdx = symengine.Derivative(f, *x)
        return cls(dfdx)

    @classmethod
    def function(cls, f: str, x) -> Expr:
        if isinstance(x, tuple):
            func = symengine.Function(f)(*x)
        else:
            func = symengine.Function(f)(x)
        return cls(func)

    @classmethod
    def piecewise(cls, *args) -> Expr:
        pw = symengine.Piecewise(*args)
        return cls(pw)

    @classmethod
    def first(cls, col, group):
        """Function giving the first value of col for all records in group"""
        return cls.function("first", (col, group))

    @classmethod
    def newind(cls):
        """The newind function

        0 - For the first record of the dataset
        1 - For the first record of each individual (except if it is the first in the dataset)
        2 - For any other record
        """
        return cls.function("newind", ())

    @classmethod
    def forward(cls, value, condition):
        """Function to carry forward value at a certain condition"""
        return cls.function("forward", (value, condition))

    def __gt__(self, other) -> BooleanExpr:
        return BooleanExpr(symengine.Gt(self._expr, other))

    def __le__(self, other) -> BooleanExpr:
        return BooleanExpr(symengine.Le(self._expr, other))


class BooleanExpr:
    # A boolean expression with all symbols real
    def __init__(self, source: TBooleanExpr):
        if isinstance(source, BooleanExpr):
            self._expr = source._expr
        else:
            self._expr = sympy.sympify(source)

    @property
    def free_symbols(self) -> set[Expr]:
        symbs = {Expr(a) for a in self._expr.free_symbols}
        return symbs

    @property
    def args(self) -> tuple[Expr, ...]:
        args = [Expr(a) for a in self._expr.args]
        return tuple(args)

    @property
    def lhs(self) -> Expr:
        lhs = self._expr.lhs
        rhs = self._expr.rhs
        # Coming from piecewise and symengine changed the order
        if isinstance(rhs, sympy.Symbol) and isinstance(lhs, sympy.Number):
            return Expr(rhs)
        else:
            return Expr(lhs)

    @property
    def rhs(self) -> Expr:
        lhs = self._expr.lhs
        rhs = self._expr.rhs
        # Coming from piecewise and symengine changed the order
        if isinstance(rhs, sympy.Symbol) and isinstance(lhs, sympy.Number):
            return Expr(lhs)
        else:
            return Expr(rhs)

    def __eq__(self, other) -> bool:
        return self._expr == other

    def __and__(self, other: BooleanExpr) -> BooleanExpr:
        return BooleanExpr(sympy.And(self._expr, other._expr))

    @classmethod
    def eq(cls, lhs: TExpr, rhs: TExpr) -> BooleanExpr:
        return cls(sympy.Eq(lhs, rhs))

    @classmethod
    def ne(cls, lhs: TExpr, rhs: TExpr) -> BooleanExpr:
        return cls(sympy.Ne(lhs, rhs))

    @classmethod
    def gt(cls, lhs: TExpr, rhs: TExpr) -> BooleanExpr:
        return cls(sympy.Gt(lhs, rhs))

    @classmethod
    def ge(cls, lhs: TExpr, rhs: TExpr) -> BooleanExpr:
        return cls(sympy.Ge(lhs, rhs))

    @classmethod
    def le(cls, lhs: TExpr, rhs: TExpr) -> BooleanExpr:
        return cls(sympy.Le(lhs, rhs))

    def unicode(self) -> str:
        return ExprPrinter().doprint(sympy.sympify(self._expr))

    def atoms(self, *types):
        return self._expr.atoms(types)

    def _symengine_(self) -> symengine.Expr:
        return symengine.sympify(self._expr)

    def _sympy_(self) -> sympy.Expr:
        return self._expr

    def __repr__(self) -> str:
        return repr(self._expr)


# Type hint for public functions taking an expression as input
TExpr = Union[int, float, str, sympy.Expr, symengine.Basic, Expr]
TSymbol = Union[str, sympy.Expr, symengine.Basic, Expr]
TBooleanExpr = Union[str, sympy.Basic, symengine.Basic, BooleanExpr]
