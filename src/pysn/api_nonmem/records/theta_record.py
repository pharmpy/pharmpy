# -*- encoding: utf-8 -*-

from collections import namedtuple

from pysn.parse_utils import AttrTree

from .parser import ThetaRecordParser
from .record import Record

ThetaInit = namedtuple('ThetaInit', ('low', 'init', 'up', 'fix', 'node'))


class ThetaRecord(Record):
    def __init__(self, buf):
        self.parser = ThetaRecordParser(buf)
        self.root = self.parser.root

    def _lexical_tokens(self):
        pass

    def ordered_pairs(self):
        pass

    def __str__(self):
        return super().__str__() + str(self.parser.root)

    @property
    def thetas(self):
        """Extracts from tree root and returns list of :class:`ThetaInit`."""

        thetas = []
        for theta in [theta for theta in self.root.all('theta')]:
            init = {k: None for k in ThetaInit._fields}
            init['node'] = theta

            for rule in ['low', 'init', 'up']:
                node = theta.find(rule)
                if node:
                    init[rule] = node.tokens[0].eval
                elif rule == 'low':
                    init[rule] = float('-INF')
                elif rule == 'up':
                    init[rule] = float('INF')
            init['fix'] = bool(theta.find('FIX'))

            node = theta.find('n')
            if node:
                n_replicate = int(node.find('INT'))
                thetas += [ThetaInit(**init) for _ in range(n_replicate)]
            else:
                thetas += [ThetaInit(**init)]

        return thetas

    @thetas.setter
    def thetas(self, tuples):
        nodes = []
        nodes_new = self._nodes_from_tuples(tuples)
        for child in self.root.children:
            if child.rule != 'theta':
                nodes += [child]
                continue
            try:
                nodes += [nodes_new.pop(0)]
            except IndexError:
                pass
        self.root = AttrTree.create('root', nodes + nodes_new)

    def _nodes_from_tuples(self, vals):
        nodes = []
        for val in vals:
            if nodes:
                nodes += [dict(WS='\n  ')]
            theta = [{'LPAR': '('}]
            if val.low is not None:
                theta += [{'low': {'NUMERIC': val.low}}]
                theta += [{'WS': ' '}]
            if val.init is not None:
                theta += [{'init': {'NUMERIC': val.init}}]
                theta += [{'WS': ' '}]
            if val.up is not None:
                theta += [{'up': {'NUMERIC': val.up}}]
            if val.fix:
                theta += [{'WS': ' '}]
                theta += [{'FIX': 'FIXED'}]
            theta += [{'RPAR': ')'}]
            nodes += [AttrTree.create('theta', theta)]
        return nodes
