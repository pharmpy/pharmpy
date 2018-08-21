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
        #self.root.treeprint()

    def option_pairs(self):
        """ Extract the key-value pairs
            If no value exists set it to None
        """
        pairs = OrderedDict()
        for keyval in self.root.all('keyval'):
            print(keyval)

    def __str__(self):
        return super().__str__() + str(self.root)
