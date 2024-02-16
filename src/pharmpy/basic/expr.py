from __future__ import annotations

from typing import Mapping

from pharmpy.deps import symengine, sympy
from pharmpy.deps.sympy_printing import pretty


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
            self._expr = symengine.sympify(source)

    @property
    def name(self) -> str:
        if isinstance(self._expr, symengine.Symbol):
            return self._expr.name
        elif isinstance(self._expr, symengine.Function):
            return self._expr.get_name()
        else:
            raise ValueError("Expression has no name")

    @property
    def args(self):
        if isinstance(self._expr, symengine.Piecewise):
            # Due to https://github.com/symengine/symengine.py/issues/469
            x = self._expr.args
            args = tuple((Expr(x[i]), BooleanExpr(x[i + 1])) for i in range(0, len(x), 2))
        else:
            args = tuple(Expr(a) for a in self._expr.args)
        return args

    def make_args(self, expr):
        return sympy.sympify(self._expr).make_args(sympy.sympify(expr))

    @property
    def free_symbols(self) -> set[Expr]:
        symbs = {Expr(a) for a in self._expr.free_symbols}
        return symbs

    def subs(self, d: Mapping[Expr, Expr]) -> Expr:
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

    def __float__(self):
        return float(self._expr)

    def __int__(self):
        return int(self._expr)

    def __bool__(self):
        return self._expr != 0

    def __repr__(self):
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

    def _sympy_(self) -> sympy.Expr:
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
        return self._expr.func == symengine.Mul

    def is_add(self) -> bool:
        return self._expr.func == symengine.Add

    def is_exp(self) -> bool:
        return self.is_pow() and self._expr.args[0] == symengine.E

    def is_pow(self) -> bool:
        return isinstance(self._expr, symengine.Pow)

    def is_function(self) -> bool:
        return self._expr.is_Function

    def is_derivative(self) -> bool:
        return self._expr.func == symengine.Derivative

    def is_piecewise(self) -> bool:
        return isinstance(self._expr, symengine.Piecewise)

    def is_nonnegative(self) -> bool | None:
        return sympy.ask(sympy.Q.nonnegative(self._expr))

    def piecewise_fold(self) -> Expr:
        if isinstance(self._expr, symengine.Piecewise):
            expr = sympy.sympify(self._expr)
            if expr.is_Piecewise:
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
    def derivative(cls, f, x) -> Expr:
        dfdx = symengine.Derivative(f, x)
        return cls(dfdx)

    @classmethod
    def function(cls, f: str, x) -> Expr:
        func = symengine.Function(f)(x)
        return cls(func)

    @classmethod
    def piecewise(cls, *args) -> Expr:
        pw = symengine.Piecewise(*args)
        return cls(pw)

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
    def args(self):
        args = [Expr(a) for a in self._expr.args]
        return tuple(args)

    @property
    def lhs(self):
        lhs = self._expr.lhs
        rhs = self._expr.rhs
        # Coming from piecewise and symengine changed the order
        if isinstance(rhs, sympy.Symbol) and isinstance(lhs, sympy.Number):
            return Expr(rhs)
        else:
            return Expr(lhs)

    @property
    def rhs(self):
        lhs = self._expr.lhs
        rhs = self._expr.rhs
        # Coming from piecewise and symengine changed the order
        if isinstance(rhs, sympy.Symbol) and isinstance(lhs, sympy.Number):
            return Expr(lhs)
        else:
            return Expr(rhs)

    @classmethod
    def eq(cls, lhs, rhs):
        return cls(sympy.Eq(lhs, rhs))

    @classmethod
    def gt(cls, lhs, rhs):
        return cls(sympy.Gt(lhs, rhs))

    @classmethod
    def ge(cls, lhs, rhs):
        return cls(sympy.Ge(lhs, rhs))

    @classmethod
    def le(cls, lhs, rhs):
        return cls(sympy.Le(lhs, rhs))

    def unicode(self):
        return ExprPrinter().doprint(sympy.sympify(self._expr))

    def atoms(self, *types):
        return self._expr.atoms(types)

    def _symengine_(self):
        return symengine.sympify(self._expr)

    def _sympy_(self):
        return self._expr

    def __repr__(self):
        return repr(self._expr)


# Type hint for public functions taking an expression as input
TExpr = int | float | str | sympy.Expr | symengine.Basic | Expr
TSymbol = str | sympy.Expr | symengine.Basic | Expr
TBooleanExpr = str | sympy.Basic | symengine.Basic | BooleanExpr
