# -*- encoding: utf-8 -*-

"""
Generic NONMEM option record class.

Assumes 'KEY=VALUE' and does not support 'KEY VALUE'.
"""

from collections import OrderedDict
from .record import Record
from .parser import OptionRecordParser


class OptionRecord(Record):
    def __init__(self, raw_text):
        self.parser = OptionRecordParser(raw_text)
        self.root = self.parser.root

    @property
    def option_pairs(self):
        """ Extract the key-value pairs
            If no value exists set it to None
        """
        pairs = OrderedDict()
        for keyval in self.root.all('pair'):
            if hasattr(keyval, 'KEY'):
                pairs[keyval.KEY] = keyval.VALUE
            else:
                pairs[keyval.VALUE] = None

        return pairs

    @property
    def first_key(self):
        """ Extract the first key
        """
        pairs = self.option_pairs
        return next(iter(pairs))

    def __str__(self):
        return super().__str__() + str(self.root)
