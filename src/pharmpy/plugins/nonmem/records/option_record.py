"""
Generic NONMEM option record class.

Assumes 'KEY=VALUE' and does not support 'KEY VALUE'.
"""

from collections import OrderedDict

from pharmpy.parse_utils.generic import AttrTree

from .record import Record


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

    def append_option(self, key, value=None):
        """ Append option as last option

            Method applicable to option records with no special grammar
        """
        if value is None:
            node = AttrTree.create('option', [{'VALUE': key}])
            self.append_option_node(node)
        else:
            raise NotImplementedError("Please implement this!")

    def append_option_node(self, node):
        """ Add a new option as last option
        """
        last_child = self.root.children[-1]
        if '\n' in str(last_child):
            if len(self.root.children) > 1 and self.root.children[-2].rule == 'ws':
                self.root.children[-1:0] = [node]
            else:
                ws_node = AttrTree.create('ws', [{'WS_ALL': ' '}])
                self.root.children[-1:0] = [ws_node, node]
        else:
            self.root.children.append(node)
