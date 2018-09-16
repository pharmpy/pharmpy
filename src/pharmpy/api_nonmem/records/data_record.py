# -*- encoding: utf-8 -*-

"""
NONMEM data record class.
"""

from .parser import DataRecordParser
from .option_record import OptionRecord
from pharmpy.input import InputFilter, InputFilters, InputFilterOperator
from pharmpy.parse_utils import AttrTree


class DataRecord(OptionRecord):
    def __init__(self, raw_text):
        self.parser = DataRecordParser(raw_text)
        self.root = self.parser.root

    @property
    def filename(self):
        """The (raw, unresolved) path of the dataset."""
        filename = self.root.filename
        if filename.find('TEXT'):
            return str(filename)
        elif filename.find('QUOTE'):
            return str(filename)[1:-1]

    @property
    def ignore_character(self):
        """The comment character from ex IGNORE=C or None if not available
        """
        if hasattr(self.root, 'ignore') and self.root.ignore.find('char'):
            return str(self.root.ignore.char)
        else:
            return None

    @property
    def filters(self):
        filters = InputFilters()

        if hasattr(self.root, 'ignore'):
            attr = 'ignore'
            filters.accept = False
        elif hasattr(self.root, 'accept'):
            attr = 'accept'
            filters.accept = True
        else:
            return filters

        for option in self.root.all(attr):
            for filt in option.all('filter'):
                symbol = filt.COLUMN
                value = filt.TEXT
                if hasattr(filt, 'OP_EQ'):
                    operator = InputFilterOperator.EQUAL
                elif hasattr(filt, 'OP_STR_EQ'):
                    operator = InputFilterOperator.STRING_EQUAL
                elif hasattr(filt, 'OP_NE'):
                    operator = InputFilterOperator.NOT_EQUAL
                elif hasattr(filt, 'OP_STR_NE'):
                    operator = InputFilterOperator.STRING_NOT_EQUAL
                elif hasattr(filt, 'OP_LT'):
                    operator = InputFilterOperator.LESS_THAN
                elif hasattr(filt, 'OP_GT'):
                    operator = InputFilterOperator.GREATER_THAN
                elif hasattr(filt, 'OP_LT_EQ'):
                    operator = InputFilterOperator.LESS_THAN_OR_EQUAL
                elif hasattr(filt, 'OP_GT_EQ'):
                    operator = InputFilterOperator.GREATER_THAN_OR_EQUAL
                filters += [InputFilter(symbol, operator, value)]
        return filters

    @filters.setter
    def filters(self, filters):
        # Remove old filters
        self.root.remove("accept")
        keep = []
        for child in self.root.children:
            if not (child.rule == 'ignore' and not hasattr(child, 'char')):
                keep.append(child)
        self.root.children = keep

        # Install new filters at the end
        if not filters:     # This was easiest kept as a special case
            return

        if filters.accept:
            tp = 'ACCEPT'
        else:
            tp = 'IGNORE'
        nodes = [{tp: tp}, {'EQUAL': '='}, {'LPAR': '('}]
        first = True
        for f in filters:
            if not first:
                nodes += [{'COMMA': ','}]
            new = [{'COLUMN': f.symbol}]
            if f.operator == InputFilterOperator.EQUAL:
                new.append({'OP_EQ': '.EQN.'})
            elif f.operator == InputFilterOperator.STRING_EQUAL:
                new.append({'OP_STR_EQ': '.EQ.'})
            elif f.operator == InputFilterOperator.NOT_EQUAL:
                new.append({'OP_NE': '.NEN.'})
            elif f.operator == InputFilterOperator.STRING_NOT_EQUAL:
                new.append({'OP_STR_NE': '.NE.'})
            elif f.operator == InputFilterOperator.LESS_THAN:
                new.append({'OP_LT': '.LT.'})
            elif f.operator == InputFilterOperator.GREATER_THAN:
                new.append({'OP_GT': '.GT.'})
            elif f.operator == InputFilterOperator.LESS_THAN_OR_EQUAL:
                new.append({'OP_LT_EQ': '.LE.'})
            elif f.operator == InputFilterOperator.GREATER_THAN_OR_EQUAL:
                new.append({'OP_GT_EQ': '.GE.'})
            new.append({'TEXT': f.value})
            nodes += [AttrTree.create('filter', new)]
            first = False
        nodes += [{'RPAR': ')'}]
        top = AttrTree.create(tp.lower(), nodes)
        self.root.children += [top]

    def __str__(self):
        return super().__str__()
