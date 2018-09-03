# -*- encoding: utf-8 -*-


# import numpy as np


class PopulationParameter:
    __slots__ = ('init', 'fix', 'lower', 'upper')

    def __init__(self, init=None, fix=None, lower=None, upper=None):
        self.init = init
        self.fix = fix
        self.lower = lower
        self.upper = upper

    def __repr__(self):
        defined = list()
        for key in self.__slots__:
            val = getattr(self, key)
            if val is not None:
                defined += [(key, val)]
        values = ', '.join('%s=%s' % (key, repr(val)) for key, val in defined)
        return '%s(%s)' % (self.__class__.__name__, values)

    def __str__(self):
        if self.fix:
            return 'fix(%s)' % (self.init,)
        if self.lower is None and self.upper is None:
            values = str(self.init)
        else:
            values = (self.lower, self.init, self.upper)
            values = ','.join('' if x is None else str(x) for x in values)
        return 'est(%s)' % (values,)


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
