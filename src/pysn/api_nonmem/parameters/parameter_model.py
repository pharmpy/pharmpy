# -*- encoding: utf-8 -*-

from pysn import generic
from pysn.parameters import ParameterList


class ParameterModel(generic.ParameterModel):
    """NONMEM 7.x implementation of :attr:`Model.parameters`, the parameters of the model."""

    @property
    def inits(self):
        params = ParameterList()
        for theta_record in self.model.get_records('THETA'):
            params += theta_record.params
        for omega_record in self.model.get_records('OMEGA'):
            params += [omega_record.matrix]
        return params
