# -*- encoding: utf-8 -*-

import numpy as np
import sympy as sym

from .record import Record
from pharmpy.parameters import Scalar
from pharmpy.parse_utils import AttrTree
from pharmpy.parameter import Parameter, ParameterSet


class ThetaRecord(Record):
    def parameters(self, first_theta):
        """Get a parameter set for this theta record.
        first_theta is the number of the first theta in this record
        """
        pset = ParameterSet()
        for theta in self.root.all('theta'):
            init = theta.init.tokens[0].eval
            fix = bool(theta.find('FIX'))
            if theta.find('low'):
                lower = theta.low.tokens[0].eval
            else:
                lower = None
            if theta.find('up'):
                upper = theta.up.tokens[0].eval
            else:
                upper = None
            multiple = theta.find('n')
            if multiple:
                n = multiple.INT
            else:
                n = 1
            for i in range(first_theta, first_theta + n):
                new_par = Parameter(f'THETA({i})', init, lower, upper, fix)
                pset.add(new_par)
        return pset


    @property
    def constraints(self):
        """An array of symbolic constraints for each theta in the record
        """
        for theta in self.root.all('theta'):
            init = theta.init.tokens[0].eval
            print("QQ", init)

    @property
    def params(self):
        inits = []
        for theta in self.root.all('theta'):
            init = {'value': theta.init.tokens[0].eval, 'fix': bool(theta.find('FIX'))}
            if theta.find('low'):
                init['lower'] = theta.low.tokens[0].eval
            if theta.find('up'):
                init['upper'] = theta.up.tokens[0].eval
            if theta.find('n'):
                inits += [Scalar(**init) for _ in range(theta.n.INT)]
            else:
                inits += [Scalar(**init)]
        return inits

    @property
    def thetas(self):
        thetas = []
        for theta in self.root.all('theta'):
            init = {'value': theta.init.tokens[0].eval, 'fix': bool(theta.find('FIX'))}
            if theta.find('low'):
                init['lower'] = theta.low.tokens[0].eval
            if theta.find('up'):
                init['upper'] = theta.up.tokens[0].eval
            if theta.find('n'):
                thetas += [Scalar(**init) for _ in range(theta.n.INT)]
            else:
                thetas += [Scalar(**init)]

        return np.array(thetas)

    @thetas.setter
    def thetas(self, thetas):
        nodes = []
        nodes_new = self._new_theta_nodes(thetas)
        for child in self.root.children:
            if child.rule != 'theta':
                nodes += [child]
                continue
            try:
                nodes += [nodes_new.pop(0)]
            except IndexError:
                pass
        self.root = AttrTree.create('root', nodes + nodes_new)

    def _new_theta_nodes(self, thetas):
        nodes = []
        for theta in thetas:
            if nodes:
                nodes += [dict(WS='\n  ')]
            new = [{'LPAR': '('}]
            if theta.lower is not None:
                new += [{'low': {'NUMERIC': theta.lower}}]
                new += [{'WS': ' '}]
            if theta.init is not None:
                new += [{'init': {'NUMERIC': theta.init}}]
                new += [{'WS': ' '}]
            if theta.upper is not None:
                new += [{'up': {'NUMERIC': theta.upper}}]
            if theta.fix:
                new += [{'WS': ' '}]
                new += [{'FIX': 'FIXED'}]
            new += [{'RPAR': ')'}]
            nodes += [AttrTree.create('theta', new)]
        return nodes
