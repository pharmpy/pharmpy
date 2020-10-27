"""
NONMEM data record class.
"""

from pharmpy.model import ModelSyntaxError
from pharmpy.parse_utils import AttrToken, AttrTree

from .option_record import OptionRecord


class DataRecord(OptionRecord):
    @property
    def filename(self):
        """The (raw, unresolved) path of the dataset."""
        filename = self.root.filename
        if filename.find('TEXT'):
            return str(filename)
        else:  # 'QUOTE'
            return str(filename)[1:-1]

    @filename.setter
    def filename(self, value):
        if not value:
            # erase and replace by * (for previous subproblem)
            new = [AttrToken('ASTERISK', '*')]
            nodes = []
            for child in self.root.children:
                if new and child.rule == 'ws':
                    nodes += [child, new.pop()]
                elif child.rule in {'ws', 'comment'}:
                    nodes += [child]
            self.root = AttrTree.create('root', nodes)
        else:
            # replace only 'filename' rule and quote appropriately if, but only if, needed
            filename = str(value)
            quoted = [
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
                node = AttrTree.create('filename', {'TEXT': filename})
            else:
                if "'" in filename:
                    node = AttrTree.create('filename', {'QUOTE': '"%s"' % filename})
                else:
                    node = AttrTree.create('filename', {'QUOTE': "'%s'" % filename})
            (pre, old, post) = self.root.partition('filename')
            self.root.children = pre + [node] + post

    @property
    def ignore_character(self):
        """The comment character from ex IGNORE=C or None if not available."""
        if hasattr(self.root, 'ignchar') and self.root.ignchar.find('char'):
            char = str(self.root.ignchar.char)
            if len(char) == 3:  # It must be quoted
                char = char[1:-1]
            return char
        else:
            return None

    @ignore_character.setter
    def ignore_character(self, c):
        if c != self.ignore_character:
            print("QQ", c, self.ignore_character)
            self.root.remove('ignchar')
            char_node = AttrTree.create('char', [{'CHAR': c}])
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
        if hasattr(self.root, 'null') and self.root.null.find('char'):
            char = str(self.root.null.char)
            if char == '+' or char == '-':
                return 0
            else:
                return float(char)
        else:
            return 0

    @property
    def ignore(self):
        filters = []
        for option in self.root.all('ignore'):
            for filt in option.all('filter'):
                filters.append(filt)
        return filters

    @ignore.deleter
    def ignore(self):
        self.root.remove('ignore')

    @property
    def accept(self):
        filters = []
        for option in self.root.all('accept'):
            for filt in option.all('filter'):
                filters.append(filt)
        return filters

    @accept.deleter
    def accept(self):
        self.root.remove('accept')

    def validate(self):
        """Syntax validation of this data record
        Assumes only on $DATA exists in this $PROBLEM.
        """
        if len(self.root.all('ignchar')) > 1:
            raise ModelSyntaxError('More than one IGNORE=c')

    # @filters.setter
    # def filters(self, filters):
    #    # Install new filters at the end
    #    if not filters:     # This was easiest kept as a special case
    #        return
    #    if filters.accept:
    #        tp = 'ACCEPT'
    #    else:
    #        tp = 'IGNORE'
    #    nodes = [{tp: tp}, {'EQUAL': '='}, {'LPAR': '('}]
    #    first = True
    #    for f in filters:
    #        if not first:
    #            nodes += [{'COMMA': ','}]
    #        new = [{'COLUMN': f.symbol}]
    #        if f.operator == InputFilterOperator.EQUAL:
    #            new.append({'OP_EQ': '.EQN.'})
    #        elif f.operator == InputFilterOperator.STRING_EQUAL:
    #            new.append({'OP_STR_EQ': '.EQ.'})
    #        elif f.operator == InputFilterOperator.NOT_EQUAL:
    #            new.append({'OP_NE': '.NEN.'})
    #        elif f.operator == InputFilterOperator.STRING_NOT_EQUAL:
    #            new.append({'OP_STR_NE': '.NE.'})
    #        elif f.operator == InputFilterOperator.LESS_THAN:
    #            new.append({'OP_LT': '.LT.'})
    #        elif f.operator == InputFilterOperator.GREATER_THAN:
    #            new.append({'OP_GT': '.GT.'})
    #        elif f.operator == InputFilterOperator.LESS_THAN_OR_EQUAL:
    #            new.append({'OP_LT_EQ': '.LE.'})
    #        elif f.operator == InputFilterOperator.GREATER_THAN_OR_EQUAL:
    #            new.append({'OP_GT_EQ': '.GE.'})
    #        new.append({'TEXT': f.value})
    #        nodes += [AttrTree.create('filter', new)]
    #        first = False
    #    nodes += [{'RPAR': ')'}]
    #    top = AttrTree.create(tp.lower(), nodes)
    #    self.root.children += [top]
