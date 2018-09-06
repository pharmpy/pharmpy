# -*- encoding: utf-8 -*-

from math import pow

import numpy as np

from pysn.api_nonmem.records.parser import OmegaRecordParser
from pysn.api_nonmem.records.record import Record
from pysn.parameter_model import Scalar, CovarianceMatrix


class OmegaRecord(Record):
    def __init__(self, buf):
        self.parser = OmegaRecordParser(buf)
        self.root = self.parser.root

    def __str__(self):
        return super().__str__() + str(self.parser.root)

    @property
    def inits(self):
        inits = []
        nreps = []
        stdevs = []
        row, col = 0, 0
        corr_coords = []
        sd_matrix = bool(self.root.find('SD'))
        corr_matrix = bool(self.root.find('CORR'))
        for i, node in enumerate(self.root.all('omega')):
            if sd_matrix or node.find('SD'):
                inits += [pow(node.init.NUMERIC, 2)]
            else:
                inits += [node.init.NUMERIC]
            if corr_matrix or node.find('CORR'):
                corr_coords += [(i, row, col)]
            nreps += [node.n.INT if node.find('n') else 1]
            if row == col:
                stdevs += inits[-1:]
                row, col = (row + 1), 0
            else:
                col += 1
        for i, row, col in corr_coords:
            inits[i] = inits[i] * stdevs[row] * stdevs[col]
        return [x for nrep, init in zip(nreps, inits) for x in [init]*nrep]

    @property
    def matrix(self):
        values = self.inits

        fix = False
        if self.root.find('FIX'):
            # TODO: FIX can be exist for individual values
            fix = True

        block = self.root.find('block')
        if block:
            size = block.find('size').tokens[0].eval
            mat = CovarianceMatrix(size)
            mat.tri = values
            if fix:
                mat.estim(var=False, covar=False)
            else:
                mat.estim(var=True, covar=True)
        else:
            if self.root.find('diagonal'):
                size = self.root.diagonal.size.INT
            else:
                size = len(values)
            mat = CovarianceMatrix(size)
            mat.diag = values
            if fix:
                mat.estim(var=False)
            else:
                mat.estim(var=True)
        return mat

    @property
    def block(self):
        values = self.inits

        fixed = False
        if self.root.find('FIX'):
            fixed = True
        # TODO: FIX can be exist for individual values

        block = self.root.find('block')
        if block:
            size = block.find('size').tokens[0].eval
            mat = np.full((size, size), Scalar(0, fix=False))
            mat[np.tril_indices(size)] = tuple(Scalar(val, fix=fixed) for val in values)
            mat[np.triu_indices(size)] = mat[np.tril_indices(size)]
        else:
            diag = self.root.find('diagonal')
            if diag:
                size = diag.find('size').tokens[0].eval
            else:
                size = len(values)
            mat = np.full((size, size), Scalar(0, fix=True))
            np.fill_diagonal(mat, tuple(Scalar(val, fix=fixed) for val in values))
        return mat
