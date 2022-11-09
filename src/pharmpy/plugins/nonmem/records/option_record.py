"""
Generic NONMEM option record class.

Assumes 'KEY=VALUE' or 'VALUE' and does not support 'KEY VALUE' in general.
"""

import re
from collections import namedtuple
from typing import Tuple

from lark import Token

from pharmpy.internals.parse import AttrToken, AttrTree

from .record import Record


def _get_key(node):
    return node.KEY


def _get_value(node):
    return getattr(node, 'VALUE', None)


Option = namedtuple('Option', ['key', 'value'])


class OptionRecord(Record):
    @property
    def option_pairs(self):
        """Extract the key-value pairs
        If no value exists set it to None
        Can only handle cases where options are supposed to be unique
        """
        return {_get_key(node): _get_value(node) for node in self.root.all('option')}

    @property
    def all_options(self):
        """Extract all options even if non-unique.
        returns a list of named two-tuples with key and value
        """
        pairs = []
        for node in self.root.all('option'):
            pairs += [Option(_get_key(node), _get_value(node))]
        return pairs

    def get_option(self, name):
        for opt in self.all_options:
            if opt.key[:3] == name[:3]:
                return opt.value
        return None

    def has_option(self, name):
        return name in self.option_pairs

    def get_option_startswith(self, s):
        for opt in self.option_pairs:
            if opt.startswith(s):
                return opt
        return None

    def get_option_lists(self, option):
        """Generator for lists of one option

        For example COMPARTMENT in $MODEL. This handles 'KEY VALUE' syntax.
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
                    yield re.split(r'\s+|,', value[1:-1])
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

        ws_token = AttrToken('WS', ' ')
        option_node = self._create_option(key, new_value)
        # If no other options add first else add just after last option
        if last_option is None:
            self.root.children = [ws_token, option_node] + self.root.children
        else:
            new_children = []
            for node in self.root.children:
                new_children.append(node)
                if node is last_option:
                    new_children += [ws_token, option_node]
            self.root.children = new_children

    def _create_option(self, key, value=None):
        if value is None:
            node = AttrTree.create('option', [{'KEY': key}])
        else:
            node = AttrTree.create('option', [{'KEY': key}, {'EQUAL': '='}, {'VALUE': value}])
        return node

    def prepend_option(self, key, value=None):
        """Prepend option"""
        node = self._create_option(key, value)
        self._prepend_option_node(node)

    def _prepend_option_node(self, node):
        """Add a new option as firt option"""
        ws_token = AttrToken('WS', ' ')
        new = [node, ws_token]
        self.root.children = [self.root.children[0]] + new + self.root.children[1:]

    def append_option(self, key, value=None):
        """Append option as last option

        Method applicable to option records with no special grammar
        """
        node = self._create_option(key, value)
        self.append_option_node(node)

    def _append_option_args(self) -> Tuple[int, int, Token]:
        children = self.root.children
        n = len(children)
        # NOTE Pop trailing whitespace if any
        j = n - 1 if children[-1].rule == 'WS' else n
        for i, child in zip(reversed(range(n)), reversed(children)):
            rule = child.rule
            if rule == 'option':
                return (i + 1, j, AttrToken('WS', ' '))
            elif rule not in ('WS', 'NEWLINE'):
                return (i + 1, j, AttrToken('NEWLINE', '\n'))

        return (0, j, AttrToken('WS', ' '))

    def append_option_node(self, node):
        """Add a new option as last option"""
        i, j, sep = self._append_option_args()
        children = self.root.children
        self.root.children = children[:i] + [sep, node] + children[i:j]

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
            if node.rule == 'option' and key == _get_key(node):
                if new_children[-1].rule == 'WS':
                    new_children.pop()
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
                    if new_children[-1].rule == 'WS':
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
        for key in self.option_pairs:
            if key.startswith(start):
                self.remove_option(key)

    @staticmethod
    def match_option(options, query):
        """Match a given option to any from a set of valid options

        NONMEM allows matching down to three letters as long as
        there are no ambiguities.

        return the canonical form of the matched option or None for no match
        """

        min_prefix_len = 3

        if len(query) < min_prefix_len:
            # NOTE This keeps the original implementation's behavior but maybe
            # this could be changed?
            return None

        i = min_prefix_len

        candidates = [option for option in options if option[:i] == query[:i]]

        while len(candidates) >= 2 and i < len(query):
            candidates = [option for option in options if option[i] == query[i]]
            i += 1

        return candidates[0] if len(candidates) == 1 else None
