# -*- encoding: utf-8 -*-

"""
NONMEM data record class.
"""

from .parser import DataRecordParser
from .option_record import OptionRecord
from pysn.generic import InputFilter, InputFilters, InputFilterOperator


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

    @property
    def ignore_character(self):
        """The comment character from ex IGNORE=C or None if not available
        """
        if hasattr(self.root, 'ignore') and self.root.ignore.find('char'):
            return str(self.root.ignore.char)
        else:
            return None

    @property
    def filters(self):
        filters = InputFilters()

        if hasattr(self.root, 'ignore'):
            for filt in self.root.ignore.all('filter'):
                symbol = filt.COLUMN
                value = filt.TEXT
                if hasattr(filt, 'OP_EQ'):
                    operator = InputFilterOperator.EQUAL
                elif hasattr(filt, 'OP_STR_EQ'):
                    operator = InputFilterOperator.STRING_EQUAL
                elif hasattr(filt, 'OP_NE'):
                    operator = InputFilterOperator.NOT_EQUAL
                elif hasattr(filt, 'OP_STR_NE'):
                    operator = InputFilterOperator.STRING_NOT_EQUAL
                elif hasattr(filt, 'OP_LT'):
                    operator = InputFilterOperator.LESS_THAN
                elif hasattr(filt, 'OP_GT'):
                    operator = InputFilterOperator.GREATER_THAN
                elif hasattr(filt, 'OP_LT_EQ'):
                    operator = InputFilterOperator.LESS_THAN_OR_EQUAL
                elif hasattr(filt, 'OP_GT_EQ'):
                    operator = InputFilterOperator.GREATER_THAN_OR_EQUAL
                filters += [ InputFilter(symbol, operator, value) ]
        return filters 


    def __str__(self):
        return super().__str__()
