"""
NONMEM data record class.
"""

from dataclasses import dataclass, replace
from functools import cached_property
from typing import Literal

from pharmpy.internals.parse import AttrToken, AttrTree
from pharmpy.model import ModelSyntaxError

from .option_record import OptionRecord, replace_tree
from .record import with_parsed_and_generated

TYPES_OF_SPACE = frozenset(('WS', 'NEWLINE'))
TYPES_OF_KEEP = frozenset(('WS', 'NEWLINE', 'COMMENT'))


@with_parsed_and_generated
@dataclass(frozen=True)
class DataRecord(OptionRecord):
    @cached_property
    def filename(self):
        """The (raw, unresolved) path of the dataset."""
        filename = self.tree.subtree('filename')
        if filename.find('FILENAME'):
            return str(filename)
        else:  # 'QFILENAME'
            return str(filename)[1:-1]

    def replace_filename(self, value):
        if not value:
            # erase and replace by * (for previous subproblem)
            new = [AttrToken('ASTERISK', '*')]
            nodes = []
            for child in self.tree.children:
                if new and child.rule in TYPES_OF_SPACE:
                    nodes += [child, new.pop()]
                elif child.rule in TYPES_OF_KEEP:
                    nodes += [child]
            new_children = nodes
            return replace_tree(self, replace(self.tree, children=new_children))

        # replace only 'filename' rule and quote appropriately if, but only if, needed
        filename = str(value)
        quoted = [
            '"',
            "'",
            ',',
            ';',
            '(',
            ')',
            '=',
            ' ',
            'IGNORE',
            'NULL',
            'ACCEPT',
            'NOWIDE',
            'WIDE',
            'CHECKOUT',
            'RECORDS',
            'RECORDS',
            'LRECL',
            'NOREWIND',
            'REWIND',
            'NOOPEN',
            'LAST20',
            'TRANSLATE',
            'BLANKOK',
            'MISDAT',
        ]
        if not any(x in filename for x in quoted):
            node = AttrTree.create('filename', {'FILENAME': filename})
        else:
            if "'" not in filename:
                node = AttrTree.create('filename', {'QFILENAME': f"'{filename}'"})
            elif '"' not in filename:
                node = AttrTree.create('filename', {'QFILENAME': f'"{filename}"'})
            else:
                raise ValueError('Cannot have both " and \' in filename.')
        pre, _, post = self.tree.partition('filename')
        new_children = pre + (node,) + post
        return replace_tree(self, replace(self.tree, children=new_children))

    @cached_property
    def ignore_character(self):
        """The comment character from ex IGNORE=C or None if not available."""
        char = None
        if self.tree.first_branch('ignchar', 'char'):
            for option in self.tree.subtrees('ignchar'):
                if char is not None:
                    raise ModelSyntaxError("Redefinition of ignore character in $DATA")
                char = str(option.subtree('char'))
                if len(char) == 3:  # It must be quoted
                    char = char[1:-1]

        return char

    def replace_ignore_character(self, c: str):
        if c == self.ignore_character:
            return self

        char = c if len(c) == 1 else f'"{c}"'
        if len(c) == 1:
            char = c
        elif "'" not in c:
            char = f"'{c}'"
        elif '"' not in c:
            char = f'"{c}"'
        else:
            raise ValueError('Cannot have both " and \' in ignore character.')
        char_node = AttrTree.create('char', [{'CHAR': char}])
        node = AttrTree.create('ignchar', [{'IGNORE': 'IGNORE'}, {'EQUALS': '='}, char_node])

        new_root = self.tree.remove('ignchar')
        return replace_tree(self, new_root).append_option_node(node)

    def replace_ignore_character_from_header(self, label):
        """Set ignore character from a header label
        If label[0] is a-zA-Z set @
        else set label[0]
        """
        c = label[0]
        return self.replace_ignore_character('@' if c.isalpha() else c)

    @cached_property
    def null_value(self):
        """The value to replace for NULL (i.e. . etc) in the dataset
        note that only +,-,0 (meaning 0) and 1-9 are allowed
        """
        if subtree := self.tree.first_branch('null', 'char'):
            char = str(subtree)
            if char == '+' or char == '-':
                return 0
            else:
                return float(char)
        else:
            return 0

    def _filters_of(self, name: Literal['ignore', 'accept']):
        for option in self.tree.subtrees(name):
            yield from option.subtrees('filter')

    @cached_property
    def ignore(self):
        return tuple(self._filters_of('ignore'))

    def del_ignore(self):
        return replace_tree(self, self.tree.remove('ignore'))

    @cached_property
    def accept(self):
        return tuple(self._filters_of('accept'))

    def del_accept(self):
        return replace_tree(self, self.tree.remove('accept'))
