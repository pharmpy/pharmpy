from pharmpy.deps import sympy
from pharmpy.internals.expr.parse import parse


class Expr:
    """A real valued symbolic expression with real symbols"""

    def __init__(self, source):
        if isinstance(source, Expr):
            self._expr = source._expr
        elif isinstance(source, sympy.Expr):
            self._expr = source
        elif isinstance(source, str):
            self._expr = parse(source)
        else:
            raise ValueError(f"Cannot create Expr from {type(source)}")

    @property
    def name(self):
        if isinstance(self._expr, sympy.Symbol):
            return self._expr.name
        else:
            raise ValueError("Expression has no name")

    def __add__(self, other):
        return Expr(self._expr + other)

    def __radd__(self, other):
        return Expr(other + self._expr)

    def __sub__(self, other):
        return Expr(self._expr - other)

    def __rsub__(self, other):
        return Expr(other - self._expr)

    def __mul__(self, other):
        return Expr(self._expr * other)

    def __rmul__(self, other):
        return Expr(other * self._expr)

    def __truediv__(self, other):
        return Expr(self._expr / other)

    def __rtruediv__(self, other):
        return Expr(other / self._expr)

    def __pow__(self, other):
        return Expr(self._expr**other)

    def __rpow__(self, other):
        return Expr(other**self._expr)

    def __neg__(self):
        return Expr(-self._expr)

    def __pos__(self):
        return Expr(+self._expr)

    def _sympy_(self):
        return self._expr

    def __eq__(self, other):
        if isinstance(other, sympy.Basic):
            return self._expr == other
        elif isinstance(other, Expr):
            return self._expr == other._expr
        else:
            return False

    def __hash__(self):
        return hash(self._expr)

    @classmethod
    def symbol(cls, name):
        symb = sympy.Symbol(name)
        return cls(symb)

    @classmethod
    def integer(cls, value):
        n = sympy.Integer(value)
        return cls(n)
