"""
Generic NONMEM option record class.

Assumes 'KEY=VALUE' and does not support 'KEY VALUE'.
"""

from collections import OrderedDict
from .record import Record
from .parsers import OptionRecordParser


class OptionRecord(Record):
    @property
    def option_pairs(self):
        """ Extract the key-value pairs
            If no value exists set it to None
        """
        pairs = OrderedDict()
        for node in self.root.all('option'):
            if hasattr(node, 'KEY'):
                pairs[node.KEY] = node.VALUE
            else:
                pairs[node.VALUE] = None

        return pairs
