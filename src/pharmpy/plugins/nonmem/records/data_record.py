"""
NONMEM data record class.
"""

from pharmpy.internals.parse import AttrToken, AttrTree
from pharmpy.model import ModelSyntaxError

from .option_record import OptionRecord

TYPES_OF_SPACE = frozenset(('WS', 'NEWLINE'))
TYPES_OF_KEEP = frozenset(('WS', 'NEWLINE', 'COMMENT'))


class DataRecord(OptionRecord):
    @property
    def filename(self):
        """The (raw, unresolved) path of the dataset."""
        filename = self.root.subtree('filename')
        if filename.find('FILENAME'):
            return str(filename)
        else:  # 'QFILENAME'
            return str(filename)[1:-1]

    @filename.setter
    def filename(self, value):
        if not value:
            # erase and replace by * (for previous subproblem)
            new = [AttrToken('ASTERISK', '*')]
            nodes = []
            for child in self.root.children:
                if new and child.rule in TYPES_OF_SPACE:
                    nodes += [child, new.pop()]
                elif child.rule in TYPES_OF_KEEP:
                    nodes += [child]
            self.root = AttrTree.create('root', nodes)
        else:
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
            (pre, _, post) = self.root.partition('filename')
            self.root = AttrTree(self.root.rule, pre + (node,) + post)

    @property
    def ignore_character(self):
        """The comment character from ex IGNORE=C or None if not available."""
        char = None
        if self.root.first_branch('ignchar', 'char'):
            for option in self.root.subtrees('ignchar'):
                if char is not None:
                    raise ModelSyntaxError("Redefinition of ignore character in $DATA")
                char = str(option.subtree('char'))
                if len(char) == 3:  # It must be quoted
                    char = char[1:-1]

        return char

    @ignore_character.setter
    def ignore_character(self, c: str):
        if c != self.ignore_character:
            self.root = self.root.remove('ignchar')
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
            self.append_option_node(node)

    def ignore_character_from_header(self, label):
        """Set ignore character from a header label
        If s[0] is a-zA-Z set @
        else set s[0]
        """
        c = label[0]
        if c.isalpha():
            self.ignore_character = '@'
        else:
            self.ignore_character = c

    @property
    def null_value(self):
        """The value to replace for NULL (i.e. . etc) in the dataset
        note that only +,-,0 (meaning 0) and 1-9 are allowed
        """
        if subtree := self.root.first_branch('null', 'char'):
            char = str(subtree)
            if char == '+' or char == '-':
                return 0
            else:
                return float(char)
        else:
            return 0

    @property
    def ignore(self):
        filters = []
        for option in self.root.subtrees('ignore'):
            for filt in option.subtrees('filter'):
                filters.append(filt)
        return filters

    @ignore.deleter
    def ignore(self):
        self.root = self.root.remove('ignore')

    @property
    def accept(self):
        filters = []
        for option in self.root.subtrees('accept'):
            for filt in option.subtrees('filter'):
                filters.append(filt)
        return filters

    @accept.deleter
    def accept(self):
        self.root = self.root.remove('accept')
