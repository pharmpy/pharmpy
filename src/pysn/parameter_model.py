# -*- encoding: utf-8 -*-


from collections import namedtuple


PopulationParameter = namedtuple('PopulationParameter', ('lower', 'init', 'upper', 'fixed'))
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
