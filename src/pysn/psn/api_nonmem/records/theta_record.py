# -*- encoding: utf-8 -*-

from .parser import ThetaRecordParser
from .record import Record


class ThetaRecord(Record):
    def __init__(self, buf):
        self.parser = ThetaRecordParser(buf)
        self.root = self.parser.root
        # self.thetas = self.parser.root.all('theta')

    def _lexical_tokens(self):
        pass

    def ordered_pairs(self):
        pass

    def __str__(self):
        return super().__str__() + str(self.parser.root)
