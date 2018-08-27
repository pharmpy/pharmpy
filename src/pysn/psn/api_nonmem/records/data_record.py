# -*- encoding: utf-8 -*-

"""
NONMEM data record class.
"""

from .parser import DataRecordParser
from .option_record import OptionRecord


class DataRecord(OptionRecord):
    def __init__(self, raw_text):
        self.parser = DataRecordParser(raw_text)
        self.root = self.parser.root

    @property
    def path(self):
        """The path of the dataset."""
        filename = self.root.filename
        if filename.find('TEXT'):
            return str(filename)
        elif filename.find('QUOTE'):
            return str(filename)[1:-1]

    def __str__(self):
        return super().__str__()
