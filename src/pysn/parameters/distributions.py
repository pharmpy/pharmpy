# -*- encoding: utf-8 -*-

import numpy as np

from .scalars import Covar
from .scalars import Scalar
from .scalars import Var


class CovarianceMatrix:
    __slots__ = ('mat')

    def __new__(cls, dim=None, data=None):
        self = super(CovarianceMatrix, cls).__new__(cls)
        if data is None:
            dim = dim if dim else 0
            self.mat = np.full((dim, dim), Scalar(0))
            self.var = Var(0)
            self.covar = Covar(0)
        else:
            self.mat = np.array([[Scalar(x) for x in row] for row in data])
            lower = np.tril_indices(self.dim, -1)
            self.var = [Var(x) for x in np.diag(self.mat)]
            self.covar = [Covar(x) for x in self.mat[lower]]
        return self

    @property
    def dim(self):
        return len(self.mat)

    @property
    def ntri(self):
        return int((self.dim**2 - self.dim)/2)

    @property
    def var(self):
        return list(np.diag(self.mat))

    @var.setter
    def var(self, value):
        try:
            value = [Var(x) for x in value]
        except TypeError:
            value = [Var(value)]*self.dim
        np.fill_diagonal(self.mat, value)

    @property
    def covar(self):
        lower = np.tril_indices(self.dim)
        return self.mat[lower]

    @covar.setter
    def covar(self, value):
        lower = np.tril_indices(self.dim, -1)
        upper = np.triu_indices(self.dim, +1)
        try:
            value = [Covar(x) for x in value]
        except TypeError:
            value = [Covar(value)]*self.ntri
        self.mat[lower] = value
        self.mat[upper] = self.mat[lower]

    @property
    def params(self):
        params = []
        for i in range(len(self.mat)):
            row = self.mat[i]
            for j in range(i+1):
                x = row[j]
                if x.fix is not None:
                    params += [x]
        return params

    @params.setter
    def params(self, value):
        try:
            value = list(value)
        except TypeError:
            var, covar = value
        else:
            row, col = 0, 0
            var, covar = [], []
            for x in value:
                if row == col:
                    var += [Var(x)]
                    row += 1
                    col = 0
                else:
                    covar += [Covar(x)]
                    col += 1
            self.mat = np.full((row, row), Scalar(0))
        self.var = var
        self.covar = covar

    def __str__(self):
        dim = self.dim
        if dim == 0:
            return '<0×0 %s>' % (self.__class__.__name__,)
        if dim == 1:
            cov = [str(self.mat[0][0])]
        else:
            cov = tuple(tuple(str(x) for x in row) for row in self.mat)
            width = tuple(max(len(x) for x in col) for col in zip(*cov))
            fmt = ' '.join('%%-%ds' % (w,) for w in width)
        lines = [fmt % row for row in cov]
        lines = self.bracket_matrix(lines)
        return '\n'.join(lines)

    def __repr__(self):
        dim = self.dim
        return '%s(dim=%d)' % (self.__class__.__name__, dim)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.dim == other.dim) and (self.mat == other.mat).all()
        else:
            return False

    @classmethod
    def bracket_matrix(cls, matrix):
        if len(matrix) == 0:
            return ['[]']
        if len(matrix) == 1:
            return ['[%s]' % str(matrix[0])]
        char = {'uleft': '⎡', 'uright': '⎤', 'left': '⎢', 'right': '⎥', 'dleft': '⎣', 'dright': '⎦'}
        out = ['%s%s%s' % (char['uleft'], matrix[0], char['uright'])]
        out += ['%s%s%s' % (char['left'], x, char['uright']) for x in matrix[1:-1]]
        out += ['%s%s%s' % (char['dleft'], matrix[-1], char['dright'])]
        return out
