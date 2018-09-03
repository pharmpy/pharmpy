# -*- encoding: utf-8 -*-


# import numpy as np


class PopulationParameter:
    __slots__ = ('init', 'fix', 'lower', 'upper')

    def __init__(self, init=0, fix=False, lower=float('-INF'), upper=float('INF')):
        self.init = init
        self.fix = fix
        self.lower = lower
        self.upper = upper

    def __eq__(self, other):
        return tuple(getattr(self, key) for key in self.__slots__) == other

    def __str__(self):
        values = tuple(getattr(self, key) for key in self.__slots__)
        values = ', '.join('%s=%s' % (key, repr(val)) for key, val in zip(self.__slots__, values))
        return '%s(%s)' % (self.__class__.__name__, values)

    def __repr__(self):
        if self.fix:
            return '<fix %s>' % (self.init,)
        values = [self.lower, self.init, self.upper]
        if self.lower == float('-INF'):
            values = values[1:]
        if self.upper == float('INF'):
            values = values[0:-1]
        return '<est %s>' % ', '.join(str(x) for x in values)


# IndividualParameter = namedtuple('IndividualParameter', ()):
# RandomVariable = namedtuple('RandomVariable', ()):


class ParameterModel:
    """Generic template for :attr:`Model.parameters`, the parameters of the model."""

    def __init__(self, model):
        self.model = model

    @property
    def population(self):
        """All population parameters."""
        raise NotImplementedError
