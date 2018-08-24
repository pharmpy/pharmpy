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

    @property
    def path(self):
        ''' Get the path to the dataset. Will remove quoting if applicable.
        '''
        if hasattr(self.root, 'non_quoted_path'):
            path = self.root.non_quoted_path.STRING
        elif hasattr(self.root, 'single_quoted_path'):
            path = str(self.root.single_quoted_path.SINGLE_QUOTED_STRING)[1:-1]
        else:
            path = str(self.root.double_quoted_path.DOUBLE_QUOTED_STRING)[1:-1]
        return path

    def __str__(self):
        return super().__str__()
