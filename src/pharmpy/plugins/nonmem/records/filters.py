from enum import Enum


# For .EQ. ... when is filter applied? Has this been investigated previously? Before converting to FDATA sometime. After applying NULL? Before checking for max 24 characters. Can an illegal string be filtered?

# Are fortran numbers fine? Function to read fortran number...

class IgnoreAccept:
    '''One IGNORE or ACCEPT from $DATA
    '''
    def __init__(self, colname, operator, value, accept=False):
        self.colname = colname
        self.operator = operator
        self.value = value
        self.accept = accept

    def __str__(self):
        return self.colname

class InputFilterOperator(Enum):
    EQUAL = 0
    NOT_EQUAL = 1
    LESS_THAN = 2
    LESS_THAN_OR_EQUAL = 3
    GREATER_THAN = 4
    GREATER_THAN_OR_EQUAL = 5
    STRING_EQUAL = 6
    STRING_NOT_EQUAL = 7

    def __str__(self):
        return self._strings[self.value]

# Install class variables. Cannot be done directly in Enum class
InputFilterOperator._strings = [ '==', '!=', '<', '<=', '>', '>=', '==', '!=' ]


class InputFilter:
    def __str__(self):
        if self.operator == InputFilterOperator.STRING_EQUAL or self.operator == InputFilterOperator.STRING_NOT_EQUAL:
            citation = '"'
        else:
            citation = ''
        return self.symbol + ' ' + str(self.operator) + ' ' + citation + str(self.value) + citation


class InputFilters(list):
    '''A list of input filters

    Attribute: accept - Set to True if this is a list of filters to accept rows
               the default is to ignore rows
    '''
    def __init__(self, *args, accept=False, **kwargs):
        self.accept = accept
        super().__init__(*args, **kwargs)

    def __str__(self):
        if self:
            expr = ' or '.join(str(e) for e in self)
            if self.accept:
                return expr
            else:
                return 'not(' + expr + ')'
        else:
            return ''

    def apply(self, data_frame, inplace=True):
        '''Apply all filters to a DataFrame
        '''
        if self:
            df = data_frame.query(str(self), inplace=inplace)
            if df is None:
                df = data_frame
            df.index = range(len(df))   # Renumber index. Don't really know how to handle index.
            return df
