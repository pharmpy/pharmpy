import sympy
import symengine
import numpy as np

from .expr import Expr


class Matrix:
    def __init__(self, source=()):
        if isinstance(source, Matrix):
            self._m = source._m
        else:
            self._m = symengine.Matrix(source)

    def __getitem__(self, ind):
        a = self._m[ind]
        if isinstance(a, symengine.DenseMatrix):
            return Matrix(a)
        else:
            return Expr(a)

    @property
    def free_symbols(self):
        return self._m.free_symbols

    def subs(self, d):
        return Matrix(self._m.subs(d))

    @property
    def rows(self):
        return self._m.rows

    @property
    def cols(self):
        return self._m.cols

    def diagonal(self):
        return Matrix(sympy.sympify(self._m).diagonal())

    def serialize(self):
        return sympy.srepr(sympy.sympify(self._m))

    def unicode(self) -> str:
        return sympy.pretty(sympy.sympify(self._m), wrap_line=False, use_unicode=True)

    def latex(self) -> str:
        expr = sympy.sympify(self._m)
        s = sympy.latex(expr, mul_symbol='dot')
        return s

    def is_zero_matrix(self) -> bool | None:
        return self._m.is_zero_matrix

    def __repr__(self):
        return self.unicode()

    def __eq__(self, other):
        return isinstance(other, Matrix) and self._m == other._m

    def __len__(self):
        return len(self._m)

    def __hash__(self):
        return hash(sympy.ImmutableMatrix(self._m))

    def __add__(self, other):
        return Matrix(self._m + other)

    def __radd__(self, other):
        return Matrix(self._m + other)

    def __matmul__(self, other):
        return Matrix(self._m @ other)

    def __rmatmul__(self, other):
        return Matrix(other._expr @ self._expr)

    def evalf(self, d):
        A = np.array(self._m.xreplace(d)).astype(np.float64)
        return A

    @classmethod
    def deserialize(cls, s):
        return cls(sympy.parse_expr(s))

    def is_positive_semidefinite(self) -> bool | None:
        isp = sympy.Matrix(self._m).is_positive_semidefinite
        return isp

    def eigenvals(self):
        d = sympy.Matrix(self._m).eigenvals()
        ud = {Expr(key): Expr(val) for key, val in d.items()}
        return ud

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_m'] = sympy.Matrix(self._m)
        return state

    def __setstate__(self, state):
        state['_m'] = symengine.Matrix(state['_m'])
        self.__dict__.update(state)

    def to_numpy(self):
        if not self._m.free_symbols:  # Not fully numeric
            a = np.array(self._m).astype(np.float64)
        else:
            raise TypeError("Symbolic matrix cannot be converted to numeric")
        return a

    def __array__(self):
        return np.array(self._m)

    def _sympy_(self):
        return sympy.sympify(self._m)
