# -*- encoding: utf-8 -*-


import numpy as np


class Scalar:
    __slots__ = ('init', 'fix', 'lower', 'upper')

    def __new__(cls, value=0, fix=False, lower=float('-INF'), upper=float('INF')):
        self = super(Scalar, cls).__new__(cls)
        try:
            self.init = value.init
            self.fix = value.fix
            self.lower = value.lower
            self.upper = value.upper
        except AttributeError:
            self.init = float(value)
            self.fix = bool(fix) if fix is not None else None
            self.lower = float(lower)
            self.upper = float(upper)
        return self

    def __str__(self):
        if self.fix is None:
            return '<val %.4G>' % (self.init,)
        elif self.fix:
            return '<fix %.4G>' % (self.init,)
        value = '%.4G' % (self.init,)
        if self.lower != float('-INF'):
            value = '%.0G<%s' % (self.lower, value)
        if self.upper != float('INF'):
            value = '%s<%0.G' % (value, self.upper)
        return '<est %s>' % (value,)

    def __repr__(self):
        args = [repr(self.init)]
        if self.fix is not None:
            args += ['fix=%s' % repr(self.fix)]
        if self.lower != float('-INF'):
            args += ['lower=%s' % repr(self.lower)]
        if self.upper != float('INF'):
            args += ['upper=%s' % repr(self.upper)]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(args))

    def __float__(self):
        return float(self.init)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all(getattr(self, a) == getattr(other, a) for a in self.__slots__)
        else:
            return False


class Var(Scalar):
    pass


class Covar(Scalar):
    pass


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


# IndividualParameter = namedtuple('IndividualParameter', ()):


# RandomVariable = namedtuple('RandomVariable', ()):


class ParameterModel:
    """Generic template for :attr:`Model.parameters`, the parameters of the model."""

    Scalar = Scalar
    CovarianceMatrix = CovarianceMatrix

    def __init__(self, model):
        self.model = model

    @property
    def population(self):
        """All population parameters."""
        raise NotImplementedError
