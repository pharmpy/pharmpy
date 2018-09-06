# -*- encoding: utf-8 -*-

import numpy as np

from pysn.api_nonmem.records.parser import ThetaRecordParser
from pysn.api_nonmem.records.record import Record
from pysn.parameter_model import Scalar
from pysn.parse_utils import AttrTree


class ThetaRecord(Record):
    def __init__(self, buf):
        self.parser = ThetaRecordParser(buf)
        self.root = self.parser.root

    def __str__(self):
        return super().__str__() + str(self.parser.root)

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
