# -*- encoding: utf-8 -*-

from pysn import generic


class ParameterModel(generic.ParameterModel):
    """NONMEM 7.x implementation of :attr:`Model.parameters`, the parameters of the model."""

    @property
    def population(self):
        params = list()
        params += [theta_rec.thetas for theta_rec in self.model.get_records('THETA')]
        params += [omega_rec.block for omega_rec in self.model.get_records('OMEGA')]
        return params
