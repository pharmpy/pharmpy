"""
Generic NONMEM option record class.

Assumes 'KEY=VALUE' or 'VALUE' and does not support 'KEY VALUE' in general.
"""

import re
from collections import namedtuple
from typing import Iterable, Optional, Tuple, Union, cast

from pharmpy.internals.parse import AttrToken, AttrTree, NoSuchRuleException
from pharmpy.internals.parse.generic import eval_token

from .record import Record


def _get_key(node: AttrTree) -> str:
    return cast(str, eval_token(node.leaf('KEY')))


def _get_value(node: AttrTree) -> Optional[str]:
    try:
        return cast(str, eval_token(node.leaf('VALUE')))
    except NoSuchRuleException:
        return None


Option = namedtuple('Option', ['key', 'value'])


class OptionRecord(Record):
    def __init__(self, name, raw_name, root):
        if hasattr(self, 'option_defs'):
            self.parsed_options = self.option_defs.parse_ast(root)
        super().__init__(name, raw_name, root)

    @property
    def option_pairs(self):
        """Extract the key-value pairs
        If no value exists set it to None
        Can only handle cases where options are supposed to be unique
        """
        return {_get_key(node): _get_value(node) for node in self.root.subtrees('option')}

    @property
    def all_options(self):
        """Extract all options even if non-unique.
        returns a list of named two-tuples with key and value
        """
        pairs = []
        for node in self.root.subtrees('option'):
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
        for node in self.root.subtrees('option'):
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
        it = iter(self.root.children)
        for node in it:
            if not isinstance(node, AttrTree) or node.rule != 'option':
                new_children.append(node)
                continue

            if _get_key(node) == key:
                new_children.append(node.replace_first(AttrToken('VALUE', new_value)))
                new_children.extend(it)
                newroot = AttrTree(self.root.rule, tuple(new_children))
                return self.__class__(self.name, self.raw_name, newroot)

            new_children.append(node)

            last_option = node

        ws_token = AttrToken('WS', ' ')
        option_node = self._create_option(key, new_value)
        # If no other options add first else add just after last option
        if last_option is None:
            newroot = AttrTree(
                self.root.rule,
                (
                    ws_token,
                    option_node,
                )
                + self.root.children,
            )
        else:
            new_children = []
            for node in self.root.children:
                new_children.append(node)
                if node is last_option:
                    new_children += [ws_token, option_node]
            newroot = AttrTree(self.root.rule, tuple(new_children))
        return self.__class__(self.name, self.raw_name, newroot)

    def _create_option(self, key: str, value: Optional[str] = None):
        if value is None:
            node = AttrTree.create('option', [{'KEY': key}])
        else:
            node = AttrTree.create('option', [{'KEY': key}, {'EQUAL': '='}, {'VALUE': value}])
        return node

    def prepend_option(self, key: str, value: Optional[str] = None):
        """Prepend option"""
        node = self._create_option(key, value)
        new_root = self._prepend_option_node(node)
        return self.__class__(self.name, self.raw_name, new_root)

    def _prepend_option_node(self, node):
        """Add a new option as first option"""
        ws_token = AttrToken('WS', ' ')
        new = (node, ws_token)
        return AttrTree(self.root.rule, self.root.children[:1] + new + self.root.children[1:])

    def append_option(self, key: str, value: Optional[str] = None):
        """Append option as last option

        Method applicable to option records with no special grammar
        """
        node = self._create_option(key, value)
        newrec = self.append_option_node(node)
        return newrec

    def _append_option_args(self) -> Tuple[int, int, AttrToken]:
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
        newroot = AttrTree(self.root.rule, children[:i] + (sep, node) + children[i:j])
        return self.__class__(self.name, self.raw_name, newroot)

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

        newroot = self.root.map(_fn)
        return self.__class__(self.name, self.raw_name, newroot)

    def remove_option(self, key):
        """Remove all options key"""
        new_children = []
        for node in self.root.children:
            if isinstance(node, AttrTree) and node.rule == 'option' and key == _get_key(node):
                if new_children[-1].rule == 'WS':
                    new_children.pop()
            else:
                new_children.append(node)
        newroot = AttrTree(self.root.rule, tuple(new_children))
        return self.__class__(self.name, self.raw_name, newroot)

    def remove_nth_option(self, key, n):
        """Remove the nth option key"""
        new_children = []
        i = 0
        for node in self.root.children:
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
        newroot = AttrTree(self.root.rule, tuple(new_children))
        return self.__class__(self.name, self.raw_name, newroot)

    def add_suboption_for_nth(self, key, n, suboption):
        """Adds a suboption to the nth option key"""
        i = 0
        new_children = []
        it = iter(self.root.children)
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
                        newroot = AttrTree(self.root.rule, tuple(new_children))
                        return self.__class__(self.name, self.raw_name, newroot)
                    i += 1

            new_children.append(node)

    def remove_suboption_for_all(self, key, suboption):
        """Remove subtoption from all options key"""
        new_children = []
        it = iter(self.root.children)
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

        newroot = AttrTree(self.root.rule, tuple(new_children))
        return self.__class__(self.name, self.raw_name, newroot)

    def remove_option_startswith(self, start):
        """Remove all options that startswith"""
        rec = self
        for key in self.option_pairs:
            if key.startswith(start):
                rec = rec.remove_option(key)
        return rec

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


class Opts:
    def __init__(self, *args):
        self.options = args
        self.find_abbreviations()

    def __getitem__(self, name):
        for opt in self.options:
            if opt.name == name:
                return opt
        raise KeyError(f"No option named {name}")

    def find_abbreviations(self):
        accepted = set()
        done = set()

        # Handle options not allowed to be abbreviated
        for opt in self.options:
            if not isinstance(opt, WildOpt) and opt.noabbrev:
                # Currenty only single options
                opt.abbreviations = [opt.name]
                accepted.add(opt.name)
                done.add(opt.name)

        # Find longest length
        longest = 0
        for opt in self.options:
            if isinstance(opt, WildOpt):
                continue
            if len(opt.name) > longest:
                longest = len(opt.name)

        curlength = longest
        while True:
            count = {}
            optdict = {}
            for opt in self.options:
                if isinstance(opt, WildOpt):
                    continue
                name = opt.name
                if name not in done and len(name) >= curlength:
                    abbrev = name[0:curlength]
                    if abbrev not in count:
                        count[abbrev] = 1
                        optdict[abbrev] = opt
                    else:
                        count[abbrev] += 1

            if len(count) == 0:
                continue

            for name, n in count.items():
                if n == 1 and name not in accepted:
                    optdict[name].abbreviations.append(name)
                    accepted.add(name)
                else:
                    done.add(name)

            if curlength == 3:
                break
            curlength -= 1

    def parse_ast(self, tree, nonoptions=None, netas=None):
        # Return a list of tuples of canonical option name, value (or None for no value)
        # Nonoptions will not be matched against options but kept as WildOpts
        opt_nodes = list(tree.subtrees('option'))
        found_options = set()
        i = 0
        parsed = []

        def add_option(opt, name, value):
            # * Same MxOpt or SimpleOpt multiple times is ignored
            # * Same opt taking argument multiple times is illegal
            # * Another option in same mx group is illegal
            if opt in found_options:
                if opt.need_value:
                    raise ValueError(f"Option {name} has already been specified.")
                else:
                    return
            elif isinstance(opt, MxOpt):
                for fopt in found_options:
                    if isinstance(fopt, MxOpt) and fopt.group == opt.group and fopt.name != name:
                        raise ValueError(
                            f"Option {fopt.name} in same group as {name} has already been specified."
                        )
            parsed.append((opt, name, value))
            found_options.add(opt)

        while i < len(opt_nodes):
            node = opt_nodes[i]
            key = _get_key(node)
            # FIXME: This is special for $TABLE. Either have special case for this or have an Opt for it.
            m = re.match(r'ETAS\(?', key)
            if m:
                # join all keys that belong with the ETAS
                etas = key
                while ')' not in key:
                    i += 1
                    node = opt_nodes[i]
                    key = _get_key(node)
                    etas += key
                etas = etas[5:-1].replace(" ", "").replace("\t", "")
                if ',' in etas:
                    a = etas.split(',')
                    for n in a:
                        parsed.append((WildOpt(), f'ETA{n}', None))
                else:
                    m = re.match(r'(?P<start>\d+)(TO|:)(?P<end>\d+|LAST)(BY(?P<by>-?\d+))?', etas)
                    if m:
                        by = 1 if m.group('by') is None else int(m.group('by'))
                        start = int(m.group('start'))
                        end = m.group('end')
                        if end == 'LAST':
                            end = netas
                        else:
                            end = int(end)
                        if start > end:
                            end -= 1
                            if by > 0:
                                by = -by
                        else:
                            if by < 0:
                                start, end = end, start
                                end -= 1
                            else:
                                end += 1
                        for n in range(start, end, by):
                            parsed.append((WildOpt(), f'ETA{n}', None))
                i += 1
                continue

            # FIXME: What happens if key=value for nonoption?
            if key in nonoptions:
                parsed.append((WildOpt(), key, None))
                i += 1
                continue
            value = _get_value(node)
            for opt in self.options:
                if opt.match(key):
                    if opt.need_value is True:
                        if value is None:
                            i += 1
                            if i < len(opt_nodes):
                                next_node = opt_nodes[i]
                                next_key = _get_key(next_node)
                                next_value = _get_value(next_node)
                                if next_value is not None:
                                    raise ValueError(f"Unexpected value for {opt.name}")
                                value = next_key
                            else:
                                raise ValueError(f"No value for option {opt.name}")
                        converted = opt.convert_value(value)
                        if converted is not None:
                            add_option(opt, opt.name, converted)
                        else:
                            raise ValueError(f"Bad value {value} for option {opt.name}")

                    elif opt.need_value is False:
                        if value is not None:
                            raise ValueError(f"Unexpected value for {opt.name}")
                        add_option(opt, opt.name, value)
                    else:  # value optional
                        parsed.append((opt, key, value))
                    break
            i += 1
        return parsed


class Opt:
    def __init__(self, noabbrev=False):
        self.abbreviations = []
        self.noabbrev = noabbrev

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def match(self, s):
        return s in self.abbreviations


class MxOpt(Opt):
    need_value = False

    def __init__(self, name, group, default=False, **kwargs):
        self.name = name
        self.group = group
        self.default = default
        super().__init__(**kwargs)


class StrOpt(Opt):
    need_value = True

    def __init__(self, name, default=None, **kwargs):
        self.name = name
        self.default = default
        super().__init__(**kwargs)

    def convert_value(self, value):
        return value


class IntOpt(Opt):
    need_value = True

    def __init__(self, name, default=None, **kwargs):
        self.name = name
        self.default = default
        super().__init__(**kwargs)

    def convert_value(self, value):
        try:
            n = int(value)
        except ValueError:
            return None
        return n


class SimpleOpt(Opt):
    need_value = False

    def __init__(self, name, **kwargs):
        self.name = name
        super().__init__(**kwargs)


class EnumOpt(Opt):
    need_value = True

    def __init__(self, name, allowed, default=None, **kwargs):
        self.name = name
        self.allowed = allowed
        if not (default is None or default in allowed):
            raise ValueError(f"Default value {default} must be in allowed: {allowed}")
        super().__init__(**kwargs)

    def convert_value(self, value):
        enum = value.upper()
        if enum in self.allowed:
            return enum
        else:
            return None


class WildOpt:
    need_value = None

    def match(self, s):
        return True

    def __eq__(self, other):
        return isinstance(other, WildOpt)
