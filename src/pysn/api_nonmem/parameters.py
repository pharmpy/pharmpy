# -*- encoding: utf-8 -*-

from pysn import generic


class ParameterModel(generic.ParameterModel):
    """A NONMEM 7.x ParameterModel implementation"""

    def initial_estimates(self, problem=0):
        pass

    def thetas(self, problem=0):
        pass
