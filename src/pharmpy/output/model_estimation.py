# -*- encoding: utf-8 -*-

from pharmpy.parameters import ParameterList


class ModelEstimation:
    """Generic template for :attr:`ModelOutput.estimation`.

    Contains estimation specifics, population/EBE results. Guided by Estimation in SO (0.3.1)."""

    def __init__(self, **named_paths):
        self.path = named_paths

    @property
    def method(self):
        raise NotImplementedError

    @property
    def population(self):
        return ParameterList()

    @property
    def individuals(self):
        return ParameterList()
