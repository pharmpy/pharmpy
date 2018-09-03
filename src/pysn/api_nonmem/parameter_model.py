# -*- encoding: utf-8 -*-

from pysn import generic


class ParameterModel(generic.ParameterModel):
    """NONMEM 7.x implementation of :attr:`Model.parameters`, the parameters of the model."""

    @property
    def population(self):
        params = []
        theta_records = [rec for rec in self.model.get_records('THETA')]
        thetas = [theta for rec in theta_records for theta in rec.thetas]
        for theta in thetas:
            params += [generic.PopulationParameter(theta.init, theta.fix, theta.low, theta.up)]
        params += [omega.block for omega in self.model.get_records('OMEGA')]
        return params
