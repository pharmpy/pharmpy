# -*- encoding: utf-8 -*-

from collections import namedtuple

from .parser import ThetaRecordParser
from .record import Record


ThetaInit = namedtuple('ThetaInit', ('lower_bound', 'init', 'upper_bound', 'fixed', 'n_thetas',
                                     'back_node'))


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
        params = [x for par in self.root.all('param') for x in par.all('single') + par.all('multi')]

        for param in params:
            init = {k: None for k in ThetaInit._fields}
            for rule in ['lower_bound', 'init', 'upper_bound']:
                node = param.find(rule)
                if node:
                    init[rule] = float(node.find('NUMERIC'))

            init['fixed'] = bool(param.find('fix'))

            node = param.find('n_thetas')
            if node:
                init['n_thetas'] = int(node.find('INT'))

            init['back_node'] = param
            thetas += [ThetaInit(**init)]

        return thetas
