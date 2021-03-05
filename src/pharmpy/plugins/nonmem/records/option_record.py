"""
Generic NONMEM option record class.

Assumes 'KEY=VALUE' and does not support 'KEY VALUE'.
"""

from collections import OrderedDict, namedtuple

from pharmpy.parse_utils.generic import AttrTree

from .record import Record


def _get_key(node):
    if hasattr(node, 'KEY'):
        return node.KEY
    else:
        return node.VALUE


def _get_value(node):
    if hasattr(node, 'KEY'):
        return node.VALUE
    else:
        return None


class OptionRecord(Record):
    @property
    def option_pairs(self):
        """Extract the key-value pairs
        If no value exists set it to None
        Can only handle cases where options are supposed to be unique
        """
        pairs = OrderedDict()
        for node in self.root.all('option'):
            pairs[_get_key(node)] = _get_value(node)
        return pairs

    @property
    def all_options(self):
        """Extract all options even if non-unique.
        returns a list of named two-tuples with key and value
        """
        Option = namedtuple('Option', ['key', 'value'])
        pairs = []
        for node in self.root.all('option'):
            pairs += [Option(_get_key(node), _get_value(node))]
        return pairs

    def get_option(self, name):
        for opt in self.all_options:
            if opt.key[0:3] == name[0:3]:
                return opt.value
        return None

    def has_option(self, name):
        return name in self.option_pairs.keys()

    def get_option_startswith(self, s):
        for opt in self.option_pairs.keys():
            if opt.startswith(s):
                return opt
        return None

    def get_option_lists(self, option):
        """Generator for lists of one option

        For example COMPARTMENT in $MODEL
        """
        next_value = False
        for node in self.root.all('option'):
            value = None
            if next_value:
                value = _get_key(node)
                next_value = False
            elif _get_key(node) == option[: len(_get_key(node))]:
                value = _get_value(node)
                if value is None:
                    next_value = True
            if value is not None:
                if value[0] == '(' and value[-1] == ')':
                    yield value[1:-1].split()
                else:
                    yield [value]

    def set_option(self, key, new_value):
        """Set the value of an option

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

    def prepend_option(self, key, value=None):
        """Prepend option"""
        node = self._create_option(key, value)
        self._prepend_option_node(node)

    def _prepend_option_node(self, node):
        """Add a new option as firt option"""
        ws_node = AttrTree.create('ws', [{'WS_ALL': ' '}])
        new = [node, ws_node]
        self.root.children = [self.root.children[0]] + new + self.root.children[1:]

    def append_option(self, key, value=None):
        """Append option as last option

        Method applicable to option records with no special grammar
        """
        node = self._create_option(key, value)
        self.append_option_node(node)

    def append_option_node(self, node):
        """Add a new option as last option"""
        last_child = self.root.children[-1]
        if last_child.rule == 'option':
            ws_node = AttrTree.create('ws', [{'WS_ALL': ' '}])
            self.root.children += [ws_node, node]
        elif last_child.rule == 'ws':
            if '\n' in str(last_child):
                ws_node = AttrTree.create('ws', [{'WS_ALL': ' '}])
                self.root.children[-1:0] = [ws_node, node]
            else:
                self.root.children.append(node)
        else:
            ws_node = AttrTree.create('ws', [{'WS_ALL': '\n'}])
            self.root.children += [ws_node, node]

    def replace_option(self, old, new):
        """Replace an option"""
        for node in self.root.all('option'):
            if hasattr(node, 'KEY'):
                if str(node.KEY) == old:
                    node.KEY = new
            elif hasattr(node, 'VALUE'):
                if str(node.VALUE) == old:
                    node.VALUE = new

    def remove_option(self, key):
        """Remove all options key"""
        new_children = []
        for node in self.root.children:
            if node.rule == 'option':
                if key == _get_key(node):
                    if new_children[-1].rule == 'ws' and '\n' not in str(new_children[-1]):
                        new_children.pop()
                else:
                    new_children.append(node)
            else:
                new_children.append(node)
        self.root.children = new_children

    def remove_nth_option(self, key, n):
        """Remove the nth option key"""
        new_children = []
        i = 0
        for node in self.root.children:
            if node.rule == 'option':
                curkey = _get_key(node)
                if key[: len(curkey)] == curkey and i == n:
                    if new_children[-1].rule == 'ws' and '\n' not in str(new_children[-1]):
                        new_children.pop()
                else:
                    new_children.append(node)
                if key[: len(curkey)] == curkey:
                    i += 1
            else:
                new_children.append(node)
        self.root.children = new_children

    def add_suboption_for_nth(self, key, n, suboption):
        """Adds a suboption to the nth option key"""
        i = 0
        for node in self.root.children:
            if node.rule == 'option':
                curkey = _get_key(node)
                if key[: len(curkey)] == curkey:
                    if i == n:
                        s = node.VALUE
                        if s.startswith('('):
                            s = f'{s[:-1]} {suboption})'
                        else:
                            s = f'({s} {suboption})'
                        node.VALUE = s
                        break
                    i += 1

    def remove_suboption_for_all(self, key, suboption):
        """Remove subtoption from all options key"""
        for node in self.root.children:
            if node.rule == 'option':
                curkey = _get_key(node)
                if key[: len(curkey)] == curkey:
                    s = node.VALUE
                    if s.startswith('('):
                        subopts = [
                            subopt
                            for subopt in s[1:-1].split()
                            if suboption[: len(subopt)] != subopt
                        ]
                    node.VALUE = '(' + ' '.join(subopts) + ')'

    def remove_option_startswith(self, start):
        """Remove all options that startswith"""
        for key in self.option_pairs.keys():
            if key.startswith(start):
                self.remove_option(key)

    @staticmethod
    def match_option(valid, option):
        """Match a given option to any from a set of valid options

        NONMEM allows matching down to three letters as long as
        there are no ambiguities.

        return the canonical form of the matched option or None for no match
        """
        i = 3
        match = None
        while i <= len(option) and not match:
            for opt in valid:
                if opt[:i] == option[:i]:
                    if match:
                        match = None
                        i += 1
                        break
                    else:
                        match = opt
            else:
                return match
        return match
