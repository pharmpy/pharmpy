# -*- encoding: utf-8 -*-

from pharmpy.parse_utils import AttrTree
from .record import Record
from .parser import ProblemRecordParser


class ProblemRecord(Record):
    def __init__(self, buf):
        self.parser = ProblemRecordParser(buf)
        self.root = self.parser.root

    @property
    def string(self):
        return str(self.root.text)

    @string.setter
    def string(self, new_str):
        if new_str != new_str.strip():
            raise ValueError("Can't set ProblemRecord.string to whitespace-padded %r" % new_str)
        node = AttrTree.create('text', dict(TEXT=new_str))
        self.root.set('text', node)

    def __str__(self):
        return super().__str__() + str(self.parser.root)
