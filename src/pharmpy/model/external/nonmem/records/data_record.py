"""
NONMEM data record class.
"""

from itertools import chain

from pharmpy.basic import BooleanExpr, Expr
from pharmpy.internals.immutable import frozenmapping
from pharmpy.internals.parse import AttrToken, AttrTree
from pharmpy.model import Ignore, ModelSyntaxError

from .option_record import OptionRecord

TYPES_OF_SPACE = frozenset(('WS', 'NEWLINE'))
TYPES_OF_KEEP = frozenset(('WS', 'NEWLINE', 'COMMENT'))
OPS = frozenmapping(
    {
        'OP_EQ': '.EQN.',
        'OP_NE': '.NEN.',
        'OP_LT': '.LT.',
        'OP_LT_EQ': '.LE.',
        'OP_GT': '.GT.',
        'OP_GT_EQ': '.GE.',
        'OP_STR_EQ': '.EQ.',
        'OP_STR_NE': '.NE.',
    }
)


class DataRecord(OptionRecord):
    @property
    def filename(self):
        """The (raw, unresolved) path of the dataset."""
        filename = self.root.subtree('filename')
        if filename.find('FILENAME'):
            return str(filename)
        else:  # 'QFILENAME'
            return str(filename)[1:-1]

    def set_filename(self, value):
        if not value:
            # erase and replace by * (for previous subproblem)
            new = [AttrToken('ASTERISK', '*')]
            nodes = []
            for child in self.root.children:
                if new and child.rule in TYPES_OF_SPACE:
                    nodes += [child, new.pop()]
                elif child.rule in TYPES_OF_KEEP:
                    nodes += [child]
            root = AttrTree.create('root', nodes)
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
            # NOTE: Filename in $DATA may be max 80 characters
            elif len(filename) > 80:
                raise ValueError(f'Filename for data record too long (>80 characters): {filename}')
            else:
                if "'" not in filename:
                    node = AttrTree.create('filename', {'QFILENAME': f"'{filename}'"})
                elif '"' not in filename:
                    node = AttrTree.create('filename', {'QFILENAME': f'"{filename}"'})
                else:
                    raise ValueError('Cannot have both " and \' in filename.')
            pre, _, post = self.root.partition('filename')
            root = AttrTree(self.root.rule, pre + (node,) + post)
        return self.replace(root=root)

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

    def set_ignore_character(self, c: str):
        if c != self.ignore_character:
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
            root = self.root.remove('ignchar')
            newrec = self.replace(root=root).append_option_node(node)
            return newrec
        else:
            return self

    def set_ignore_character_from_header(self, label):
        """Set ignore character from a header label
        If s[0] is a-zA-Z set @
        else set s[0]
        """
        c = label[0]
        if c.isalpha():
            return self.set_ignore_character('@')
        else:
            return self.set_ignore_character(c)

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

    def get_selects(self, ignore: bool) -> list[Ignore]:
        filters = self.ignore if ignore else self.accept
        selects = []
        for f in filters:
            col = f.find('COLUMN').value
            try:
                expr = f.find('EXPR').value
            except AttributeError:
                expr = f.find('QEXPR').value
            assert expr is not None
            op = [tok for tok in f.tokens if tok.rule in OPS.keys()]
            assert len(op) == 1
            op = op[0].rule
            lhs = Expr.symbol(col)
            expr = Expr(expr) if expr.isnumeric() else expr
            if 'STR' in op:
                rhs = Expr.symbol("S")
                strings = {rhs: str(expr)}
            else:
                rhs = expr
                strings = {}
            if op in {'OP_EQ', 'OP_STR_EQ'}:
                bexp = BooleanExpr.eq(lhs, rhs)
            elif op in {'OP_NE', 'OP_STR_NE'}:
                bexp = BooleanExpr.ne(lhs, rhs)
            elif op == 'OP_LT':
                bexp = BooleanExpr.lt(lhs, rhs)
            elif op == 'OP_GT':
                bexp = BooleanExpr.gt(lhs, rhs)
            elif op == 'OP_LT_EQ':
                bexp = BooleanExpr.le(lhs, rhs)
            else:  # op == 'OP_GT_EQ'
                bexp = BooleanExpr.ge(lhs, rhs)
            if not ignore:
                bexp = ~bexp
            sel = Ignore.create(expression=bexp, strings=strings)
            selects.append(sel)
        return selects

    def remove_ignore(self):
        newroot = self.root.remove('ignore')
        return self.replace(root=newroot)

    @property
    def accept(self):
        filters = []
        for option in self.root.subtrees('accept'):
            for filt in option.subtrees('filter'):
                filters.append(filt)
        return filters

    def remove_accept(self):
        newroot = self.root.remove('accept')
        return self.replace(root=newroot)

    def add_ignore(self, new):
        old = self.get_selects(ignore=True) + self.get_selects(ignore=False)

        ignore_token = AttrToken('IGNORE', 'IGNORE')
        eq_token = AttrToken('EQUALS', '=')
        lpar_token = AttrToken('LPAR', '(')
        rpar_token = AttrToken('RPAR', ')')

        nodes = []
        for ignore in new:
            if ignore in old:
                continue
            expr, strings = ignore.expression, ignore.strings
            rhs = expr.rhs if not strings else strings[expr.rhs]

            op = 'OP_'
            if strings:
                op += 'STR_'
            if expr.is_eq():
                op += 'EQ'
            elif expr.is_ne():
                op += 'NE'
            elif expr.is_lt():
                op += 'LT'
            elif expr.is_gt():
                op += 'GT'
            elif expr.is_le():
                op += 'LT_EQ'
            else:  # expr.is_ge()
                op += 'GT_EQ'

            ignore_tree = AttrTree.create(
                'filter', [{'COLUMN': str(expr.lhs), op: OPS[op], 'EXPR': str(rhs)}]
            )

            node = AttrTree('ignore', (ignore_token, eq_token, lpar_token, ignore_tree, rpar_token))
            nodes.append(node)

        if not nodes:
            return self

        children = self.root.children
        newroot = AttrTree(self.root.rule, children + self._insert_whitespace(nodes))

        return self.replace(root=newroot)

    @staticmethod
    def _insert_whitespace(nodes):
        ws_token = AttrToken('WS', ' ')
        return (
            (ws_token,)
            + tuple(chain.from_iterable((node, ws_token) for node in nodes[:-1]))
            + (nodes[-1],)
        )
