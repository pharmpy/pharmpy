from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Union, overload

from pharmpy.deps import numpy as np
from pharmpy.deps import symengine, sympy

from .expr import Expr


class Matrix:
    def __init__(self, source: Union[sympy.Matrix, symengine.Matrix, Matrix, Iterable] = ()):
        if isinstance(source, Matrix):
            self._m = source._m
        else:
            self._m = symengine.Matrix(source)

    @overload
    def __getitem__(self, ind: tuple[int, int]) -> Expr: ...

    @overload
    def __getitem__(
        self,
        ind: Union[
            tuple[Sequence, Sequence],
            tuple[int, Sequence],
            tuple[Sequence, int],
            tuple[slice, int],
            tuple[int, slice],
            tuple[slice, Sequence],
            tuple[Sequence, slice],
        ],
    ) -> Matrix: ...

    @overload
    def __getitem__(self, ind: int) -> Expr: ...

    def __getitem__(self, ind) -> Union[Expr, Matrix]:
        a = self._m[ind]
        if isinstance(a, symengine.DenseMatrix):
            return Matrix(a)
        else:
            return Expr(a)

    def __iter__(self):
        for row in range(self.rows):
            for col in range(self.cols):
                yield self[row, col]

    @property
    def free_symbols(self) -> set[Expr]:
        return self._m.free_symbols

    def subs(self, d: Mapping) -> Matrix:
        return Matrix(self._m.subs(d))

    @property
    def rows(self) -> int:
        return self._m.rows

    @property
    def cols(self) -> int:
        return self._m.cols

    def diagonal(self) -> Matrix:
        return Matrix(sympy.sympify(self._m).diagonal())

    def serialize(self) -> str:
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

    def __add__(self, other) -> Matrix:
        other = self._convert_input(other)
        if not isinstance(other, Matrix):
            return NotImplemented
        return Matrix(self._m + other)

    def __radd__(self, other) -> Matrix:
        other = self._convert_input(other)
        if not isinstance(other, Matrix):
            return NotImplemented
        return Matrix(self._m + other)

    def __matmul__(self, other) -> Matrix:
        other = self._convert_input(other)
        if not isinstance(other, Matrix):
            return NotImplemented
        return Matrix(self._m @ other)

    def __rmatmul__(self, other) -> Matrix:
        other = self._convert_input(other)
        if not isinstance(other, Matrix):
            return NotImplemented
        return Matrix(other._m @ self._m)

    @staticmethod
    def _convert_input(m):
        if isinstance(m, Matrix):
            return m
        elif isinstance(m, Sequence) and isinstance(m[0], Sequence):
            return Matrix(m)
        else:
            return m

    def evalf(self, d: Mapping) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message="__array__ implementation doesn't accept a copy keyword",
            )
            A = np.array(self._m.xreplace(d)).astype(np.float64)
        return A

    @classmethod
    def deserialize(cls, s) -> Matrix:
        return cls(sympy.parse_expr(s))

    def is_positive_semidefinite(self) -> bool | None:
        isp = sympy.Matrix(self._m).is_positive_semidefinite
        return isp

    def eigenvals(self) -> dict[Expr, Expr]:
        d = sympy.Matrix(self._m).eigenvals()
        assert isinstance(d, dict)
        ud = {Expr(key): Expr(val) for key, val in d.items()}
        return ud

    def cholesky(self) -> Matrix:
        return Matrix(self._m.cholesky())

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_m'] = sympy.Matrix(self._m)
        return state

    def __setstate__(self, state):
        state['_m'] = symengine.Matrix(state['_m'])
        self.__dict__.update(state)

    def to_numpy(self) -> np.ndarray:
        if not self._m.free_symbols:  # Not fully numeric
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=DeprecationWarning,
                    message="__array__ implementation doesn't accept a copy keyword",
                )
                a = np.array(self._m).astype(np.float64)
        else:
            raise TypeError("Symbolic matrix cannot be converted to numeric")
        return a

    def __array__(self) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message="__array__ implementation doesn't accept a copy keyword",
            )
            a = np.array(self._m)
        return a

    def _sympy_(self) -> sympy.Expr:
        return sympy.sympify(self._m)

    def _symengine_(self) -> symengine.Expr:
        return self._m
