# -*- encoding: utf-8 -*-

from pharmpy.parameters import ParameterList
from pharmpy import generic


class ModelEstimation(generic.ModelEstimation):
    """A NONMEM 7.x model estimation class."""

    @property
    def method(self):
        raise NotImplementedError

    @property
    def population(self):
        return ParameterList()

    @property
    def individuals(self):
        return ParameterList()
