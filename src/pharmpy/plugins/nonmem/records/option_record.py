"""
Generic NONMEM option record class.

Assumes 'KEY=VALUE' and does not support 'KEY VALUE'.
"""

from collections import OrderedDict, namedtuple

from pharmpy.parse_utils.generic import AttrTree

from .record import Record


class OptionRecord(Record):
    @property
    def option_pairs(self):
        """ Extract the key-value pairs
            If no value exists set it to None
            Can only handle cases where options are supposed to be unique
        """
        pairs = OrderedDict()
        for node in self.root.all('option'):
            if hasattr(node, 'KEY'):
                pairs[node.KEY] = node.VALUE
            else:
                pairs[node.VALUE] = None
        return pairs

    @property
    def all_options(self):
        """ Extract all options even if non-unique.
            returns a list of named two-tuples with key and value
        """
        Option = namedtuple('Option', ['key', 'value'])
        pairs = []
        for node in self.root.all('option'):
            if hasattr(node, 'KEY'):
                pairs += [Option(node.KEY, node.VALUE)]
            else:
                pairs += [Option(node.VALUE, None)]
        return pairs

    def set_option(self, key, new_value):
        """ Set the value of an option

            If option already exists replaces its value
            appends option at the end if it does not exist
            does not handle abbreviations yet
        """
        # If already exists update value
        last_option = None
        for node in self.root.all('option'):
            old_key = node.find('KEY')
            if str(old_key) == key:
                node.VALUE = new_value
                return
            last_option = node

        ws_node = AttrTree.create('ws', [{'WS_ALL': ' '}])
        option_node = self._create_option(key, new_value)
        # If no other options add first else add just after last option
        if last_option is None:
            self.root.children = [ws_node, option_node] + self.root.children
        else:
            new_children = []
            for node in self.root.children:
                new_children.append(node)
                if node is last_option:
                    new_children += [ws_node, option_node]
            self.root.children = new_children

    def _create_option(self, key, value=None):
        if value is None:
            node = AttrTree.create('option', [{'VALUE': key}])
        else:
            node = AttrTree.create('option', [{'KEY': key}, {'EQUAL': '='}, {'VALUE': value}])
        return node

    def append_option(self, key, value=None):
        """ Append option as last option

            Method applicable to option records with no special grammar
        """
        node = self._create_option(key, value)
        self.append_option_node(node)

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
