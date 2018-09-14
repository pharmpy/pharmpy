from enum import Enum

class InputFilterOperator(Enum):
    EQUAL = 1
    NOT_EQUAL = 2
    LESS_THAN = 3
    LESS_THAN_OR_EQUAL = 4
    GREATER_THAN = 5
    GREATER_THAN_OR_EQUAL = 6
    STRING_EQUAL = 7
    STRING_NOT_EQUAL = 8

    def __str__(self):
        if self.operator == InputFilterOperator.EQUAL or self.operator == InputFilterOperator.STRING_EQUAL:
            return "=="
        elif self.operator == InputFilterOperator.NOT_EQUAL or self.operator == InputFilterOperator.STRING_NOT_EQUAL:
            return "!="
        elif self.operator == InputFilterOperator.LESS_THAN:
            return "<"
        elif self.operator == InputFilterOperator.LESS_THAN_OR_EQUAL:
            return "<="
        elif self.operator == InputFilterOperator.GREATER_THAN:
            return ">"
        elif self.operator == InputFilterOperator.GREATER_THAN_OR_EQUAL:
            return ">="


class InputFilter:
    def __init__(self, symbol, operator, value):
        self.symbol = symbol
        self.operator = operator
        self.value = value

    def __str__(self):
        if self.operator == InputFilterOperator.STRING_EQUAL or self.operator == InputFilterOperator.STRING_NOT_EQUAL:
            citation = '"'
        else:
            citation = ''
        return self.symbol + ' ' + str(self.operator) + citation + self.value + citation

    def apply(self, data_frame, inplace=True):
        '''Apply this filter to a DataFrame.

        Arguments:
            data_frame: The data frame.
        '''
        df = data_frame.query(str(self), inplace=inplace)
        return df


class InputFilters(list):
    def apply(self, data_frame, inplace=True):
        '''Apply filters to a DataFrame
        '''
        if inplace:
            for filt in self:
                filt.apply(data_frame, inplace)
            return None
        else:
            df = data_frame
            for filt in self:
                df = filt.apply(df, inplace)
            return df
