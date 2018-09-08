# -*- encoding: utf-8 -*-

from pysn.parameters import ParameterList
from pysn import generic


class ModelEstimation(generic.output.ModelEstimation):
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
