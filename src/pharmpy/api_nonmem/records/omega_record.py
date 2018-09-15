# -*- encoding: utf-8 -*-

from math import pow

from pharmpy.api_nonmem.records.parser import OmegaRecordParser
from pharmpy.api_nonmem.records.record import Record
from pharmpy.parameters import CovarianceMatrix
from pharmpy.parameters import Scalar


class OmegaRecord(Record):
    def __init__(self, buf):
        self.parser = OmegaRecordParser(buf)
        self.root = self.parser.root

    def __str__(self):
        return super().__str__() + str(self.parser.root)

    @property
    def params(self):
        scalar_args, nreps, stdevs = [], [], []

        row, col = 0, 0
        corr_coords = []
        fixed = bool(self.root.find('FIX'))
        sd_matrix = bool(self.root.find('SD'))
        corr_matrix = bool(self.root.find('CORR'))
        for i, node in enumerate(self.root.all('omega')):
            init = node.init.NUMERIC
            if sd_matrix or node.find('SD'):
                init = pow(init, 2)
            if fixed or bool(node.find('FIX')):
                scalar_args += [(init, True)]
            else:
                scalar_args += [(init, False)]
            if corr_matrix or node.find('CORR'):
                corr_coords += [(i, row, col)]
            nreps += [node.n.INT if node.find('n') else 1]
            if row == col:
                stdevs += [init]
                row, col = (row + 1), 0
            else:
                col += 1

        for i, row, col in corr_coords:
            scalar_args[i][0] = scalar_args[i][0] * stdevs[row] * stdevs[col]

        params = [Scalar(*a) for N, args in zip(nreps, scalar_args) for a in [args]*N]
        return params

    @property
    def matrix(self):
        params = self.params
        block = self.root.find('block')
        if block:
            size = block.find('size').tokens[0].eval
            mat = CovarianceMatrix(size)
            mat.params = params
        else:
            if self.root.find('diagonal'):
                size = self.root.diagonal.size.INT
            else:
                size = len(params)
            mat = CovarianceMatrix(size)
            mat.var = params
            mat.covar = Scalar(0, fix=None)
        return mat
