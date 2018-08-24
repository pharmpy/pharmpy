# -*- encoding: utf-8 -*-

"""
NONMEM data record class.
"""

from .record import Record
from .parser import DataRecordParser
from .option_record import OptionRecord

class DataRecord(OptionRecord):
    def __init__(self, raw_text):
        self.parser = DataRecordParser(raw_text)
        self.root = self.parser.root

    def __str__(self):
        return super().__str__()
