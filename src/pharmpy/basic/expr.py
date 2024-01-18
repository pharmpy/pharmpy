from pharmpy.deps import symengine, sympy


class Expr:
    """A real valued symbolic expression with real symbols"""

    def __init__(self, source):
        if isinstance(source, Expr):
            self._expr = source._expr
        else:
            self._expr = symengine.sympify(source)

    @property
    def name(self):
        if isinstance(self._expr, symengine.Symbol):
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
        return Expr(self._expr)

    def _sympy_(self):
        return sympy.sympify(self._expr)

    def __eq__(self, other):
        return self._expr == other

    def __hash__(self):
        return hash(self._expr)

    @classmethod
    def symbol(cls, name):
        symb = symengine.Symbol(name)
        return cls(symb)

    @classmethod
    def integer(cls, value):
        n = symengine.Integer(value)
        return cls(n)
