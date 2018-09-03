# -*- encoding: utf-8 -*-

from math import pow

import numpy as np

from pysn.api_nonmem.records.parser import OmegaRecordParser
from pysn.api_nonmem.records.record import Record
from pysn.parameter_model import PopulationParameter as Param


class OmegaRecord(Record):
    def __init__(self, buf):
        self.parser = OmegaRecordParser(buf)
        self.root = self.parser.root

    def __str__(self):
        return super().__str__() + str(self.parser.root)

    @property
    def block(self):
        nodes = self.root.all('omega')

        values = [node.init.NUMERIC for node in nodes]
        if self.root.find('FIX'):
            fixed = [True]*len(nodes)
        else:
            fixed = [bool(node.find('FIX')) for node in nodes]
        if self.root.find('SD'):
            sd = [not node.find('VAR') for node in nodes]
        else:
            sd = [node.find('SD') for node in nodes]

        block = self.root.find('block')
        if block:
            size = block.find('size').tokens[0].eval
            mat = np.full((size, size), Param(0, fix=False))
            diag_idx = []
            for idx in range(size):
                diag_idx += [idx + sum(diag_idx)]
            for idx in range(len(values)):
                if idx in diag_idx and nodes[idx].find('SD'):
                    values[idx] = pow(values[idx], 2)
            mat[np.tril_indices(size)] = tuple(Param(val, fix) for val, fix in zip(values, fixed))
            mat[np.triu_indices(size)] = mat[np.tril_indices(size)]
        else:
            diag = self.root.find('diagonal')
            if diag:
                size = diag.find('size').tokens[0].eval
            else:
                size = len(values)
            for idx, sd in enumerate(node.find('SD') for node in nodes):
                values[idx] = pow(values[idx], 2) if sd else values[idx]
            mat = np.full((size, size), Param(0, fix=True))
            np.fill_diagonal(mat, tuple(Param(val, fix) for val, fix in zip(values, fixed)))
        return mat

    @block.setter
    def block(self, mat):
        pass
