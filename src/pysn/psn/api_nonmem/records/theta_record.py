import re
from enum import Enum
from collections import deque

from .record import Record
from .lexers import ThetaRecordLexer


class ThetaRecord(Record):
    def __init__(self, string):
        self.lexer = ThetaRecordLexer(string)

    def _lexical_tokens(self):
        pass

    def ordered_pairs(self):
        pass

    def __str__(self):
        result = ""
        for token in self.lexer.tokens:
            result += token.content
        return super().__str__() + result
