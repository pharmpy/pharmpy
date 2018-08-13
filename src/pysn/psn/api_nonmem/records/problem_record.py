# -*- encoding: utf-8 -*-

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
        assert new_str == new_str.strip()
        self.root.text.set(new_str)

    def __str__(self):
        return super().__str__() + str(self.parser.root)
