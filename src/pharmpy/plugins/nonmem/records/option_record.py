"""
Generic NONMEM option record class.

Assumes 'KEY=VALUE' or 'VALUE' and does not support 'KEY VALUE' in general.
"""

import re
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Iterable, Mapping, Optional, Tuple, Union, cast

from pharmpy.internals.immutable import frozenmapping
from pharmpy.internals.parse import AttrToken, AttrTree, NoSuchRuleException
from pharmpy.internals.parse.generic import eval_token

from .record import ReplaceableRecord, replace_tree, with_parsed_and_generated


def _get_key(node: AttrTree) -> str:
    return cast(str, eval_token(node.leaf('KEY')))


def _get_value(node: AttrTree) -> Optional[str]:
    try:
        return cast(str, eval_token(node.leaf('VALUE')))
    except NoSuchRuleException:
        return None


@dataclass(frozen=True)
class Option:
    key: str
    value: Optional[str]


@with_parsed_and_generated
@dataclass(frozen=True)
class OptionRecord(ReplaceableRecord):
    @cached_property
    def option_pairs(self) -> Mapping[str, Optional[str]]:
        """Extract the key-value pairs
        If no value exists set it to None
        Can only handle cases where options are supposed to be unique
        """
        return frozenmapping(
            (_get_key(node), _get_value(node)) for node in self.tree.subtrees('option')
        )

    @cached_property
    def all_options(self) -> Tuple[Option]:
        """Extract all options even if non-unique.
        returns a list of named two-tuples with key and value
        """
        return tuple(
            Option(_get_key(node), _get_value(node)) for node in self.tree.subtrees('option')
        )

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
        for node in self.tree.subtrees('option'):
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

    def set_option(self, key: str, new_value: str):
        """Set the value of an option

        If option already exists replaces its value
        appends option at the end if it does not exist
        does not handle abbreviations yet
        """
        # If already exists update value
        last_option = None
        new_children = []
        it = iter(self.tree.children)
        for node in it:
            if not isinstance(node, AttrTree) or node.rule != 'option':
                new_children.append(node)
                continue

            if _get_key(node) == key:
                new_children.append(node.replace_first(AttrToken('VALUE', new_value)))
                new_children.extend(it)
                return replace_tree(self, AttrTree(self.tree.rule, tuple(new_children)))

            new_children.append(node)

            last_option = node

        ws_token = AttrToken('WS', ' ')
        option_node = self._create_option(key, new_value)
        # If no other options add first else add just after last option
        if last_option is None:
            return replace_tree(
                self,
                replace(
                    self.tree,
                    children=(
                        ws_token,
                        option_node,
                    )
                    + self.tree.children,
                ),
            )
        else:
            new_children = []
            for node in self.tree.children:
                new_children.append(node)
                if node is last_option:
                    new_children += [ws_token, option_node]

            return replace_tree(self, replace(self.tree, children=tuple(new_children)))

    def _create_option(self, key: str, value: Optional[str] = None):
        return (
            AttrTree.create('option', [{'KEY': key}])
            if value is None
            else AttrTree.create('option', [{'KEY': key}, {'EQUAL': '='}, {'VALUE': value}])
        )

    def prepend_option(self, key: str, value: Optional[str] = None):
        """Prepend option"""
        node = self._create_option(key, value)
        return self._prepend_option_node(node)

    def _prepend_option_node(self, node):
        """Add a new option as firt option"""
        ws_token = AttrToken('WS', ' ')
        to_insert = (node, ws_token)
        new_children = self.tree.children[:1] + to_insert + self.tree.children[1:]
        return replace_tree(self, replace(self.tree, children=new_children))

    def append_option(self, key: str, value: Optional[str] = None):
        """Append option as last option

        Method applicable to option records with no special grammar
        """
        node = self._create_option(key, value)
        return self.append_option_node(node)

    def _append_option_args(self) -> Tuple[int, int, AttrToken]:
        children = self.tree.children
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
        children = self.tree.children
        new_children = children[:i] + (sep, node) + children[i:j]
        return replace_tree(self, replace(self.tree, children=new_children))

    def replace_option(self, old, new):
        """Replace an option"""

        def _fn(node: Union[AttrTree, AttrToken]):
            if isinstance(node, AttrTree) and node.rule == 'option':
                if node.find('KEY') is not None:
                    if eval_token(node.leaf('KEY')) == old:
                        return node.replace_first(AttrToken('KEY', new))
                elif node.find('VALUE') is not None:
                    if eval_token(node.leaf('VALUE')) == old:
                        return node.replace_first(AttrToken('VALUE', new))

            return node

        return replace_tree(self, self.tree.map(_fn))

    def remove_option(self, key):
        """Remove all options key"""
        new_children = []
        for node in self.tree.children:
            if isinstance(node, AttrTree) and node.rule == 'option' and key == _get_key(node):
                if new_children[-1].rule == 'WS':
                    new_children.pop()
            else:
                new_children.append(node)

        return replace_tree(self, replace(self.tree, children=tuple(new_children)))

    def remove_nth_option(self, key, n):
        """Remove the nth option key"""
        new_children = []
        i = 0
        for node in self.tree.children:
            if isinstance(node, AttrTree) and node.rule == 'option':
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

        return replace_tree(self, replace(self.tree, children=tuple(new_children)))

    def add_suboption_for_nth(self, key, n, suboption):
        """Adds a suboption to the nth option key"""
        i = 0
        new_children = []
        it = iter(self.tree.children)
        for node in it:
            if isinstance(node, AttrTree) and node.rule == 'option':
                curkey = _get_key(node)
                if key[: len(curkey)] == curkey:
                    if i == n:
                        s = node.leaf('VALUE').value
                        if s.startswith('('):
                            s = f'{s[:-1]} {suboption})'
                        else:
                            s = f'({s} {suboption})'
                        new_children.append(node.replace_first(AttrToken('VALUE', s)))
                        new_children.extend(it)
                        return replace_tree(self, replace(self.tree, children=tuple(new_children)))
                    i += 1

            new_children.append(node)

        raise ValueError('Could not find nth option key')

    def remove_suboption_for_all(self, key, suboption):
        """Remove subtoption from all options key"""
        new_children = []
        it = iter(self.tree.children)
        for node in it:
            if isinstance(node, AttrTree) and node.rule == 'option':
                curkey = _get_key(node)
                if key[: len(curkey)] == curkey:
                    s = node.leaf('VALUE').value
                    if s.startswith('('):
                        subopts = [
                            subopt
                            for subopt in s[1:-1].split()
                            if suboption[: len(subopt)] != subopt
                        ]
                        s = '(' + ' '.join(subopts) + ')'
                        new_children.append(node.replace_first(AttrToken('VALUE', s)))
                        continue

            new_children.append(node)

        return replace_tree(self, replace(self.tree, children=tuple(new_children)))

    def remove_option_startswith(self, start):
        """Remove all options that startswith"""
        current = self
        for key in self.option_pairs:
            if key.startswith(start):
                current = current.remove_option(key)

        return current

    @staticmethod
    def match_option(options: Iterable[str], query: str):
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

        i: int = min_prefix_len

        candidates = [option for option in options if option[:i] == query[:i]]

        while len(candidates) >= 2 and i < len(query):
            candidates = [
                option for option in candidates if i < len(option) and option[i] == query[i]
            ]
            i += 1

        return candidates[0] if len(candidates) == 1 else None
