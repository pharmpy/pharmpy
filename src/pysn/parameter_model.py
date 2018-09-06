# -*- encoding: utf-8 -*-


import numpy as np


class PopulationParameter:
    pass


class Scalar(PopulationParameter):
    __slots__ = ('init', 'fix', 'lower', 'upper')

    def __new__(cls, init=0, fix=None, lower=float('-INF'), upper=float('INF')):
        self = super(PopulationParameter, cls).__new__(cls)
        self.init = float(init)
        self.fix = bool(fix) if fix is not None else None
        self.lower = float(lower)
        self.upper = float(upper)
        return self

    def __eq__(self, other):
        return tuple(getattr(self, key) for key in self.__slots__) == other

    def __str__(self):
        values = tuple(getattr(self, key) for key in self.__slots__)
        values = ', '.join('%s=%s' % (key, repr(val)) for key, val in zip(self.__slots__, values))
        return '%s(%s)' % (self.__class__.__name__, values)

    def __repr__(self):
        if self.fix is None:
            return '<val %s>' % (self.init,)
        elif self.fix:
            return '<fix %s>' % (self.init,)
        values = [self.lower, self.init, self.upper]
        if self.lower == float('-INF'):
            values = values[1:]
        if self.upper == float('INF'):
            values = values[0:-1]
        return '<est %s>' % ', '.join(str(x) for x in values)

    def __float__(self):
        return float(self.init)


class CovarianceMatrix(PopulationParameter):
    __slots__ = ('cov')

    def __new__(cls, size=None, cov=None):
        self = super(CovarianceMatrix, cls).__new__(cls)
        if size:
            self.cov = np.full((size, size), Scalar(0))
        elif cov is not None:
            self.cov = np.asanyarray([[Scalar(x) for x in row] for row in cov])
        return self

    @property
    def size(self):
        return len(self.cov)

    @property
    def var(self):
        return list(np.var())

    @var.setter
    def var(self, values):
        np.fill_diagonal(self.cov, [Scalar(x) for x in values])

    @property
    def covar(self):
        return self.cov[np.tril_indices(self.size)]

    @covar.setter
    def covar(self, values):
        self.cov[np.tril_indices(self.size)] = [Scalar(x) for x in values]
        self.cov[np.triu_indices(self.size)] = self.cov[np.tril_indices(self.size)]

    @property
    def params(self):
        params = []
        for par in [val for row in self.cov for val in row]:
            if par.fix is not None:
                params += [par]
        return params

    def estim(self, var=None, covar=None):
        for i in range(self.size):
            for j in range(self.size):
                if i == j and var is not None:
                    self.cov[i][j].fix = not var
                elif i != j and covar is not None:
                    self.cov[i][j].fix = not covar

    def __str__(self):
        dim = self.size
        if dim == 0:
            return '<0×0 %s>' % (self.__class__.__name__,)
        if dim == 1:
            cov = [repr(self.cov[0][0])]
        else:
            cov = tuple(tuple(repr(x) for x in row) for row in self.cov)
            width = tuple(max(len(x) for x in col) for col in zip(*cov))
            fmt = ' '.join('%%-%ds' % (w,) for w in width)
        lines = [fmt % row for row in cov]
        lines = self.bracket_matrix(lines)
        return '\n'.join(lines)

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
