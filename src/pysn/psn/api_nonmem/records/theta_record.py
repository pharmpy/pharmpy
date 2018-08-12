import re
from enum import Enum
from collections import deque

from .record import Record
from .lexers import ThetaRecordLexer
from .parser import ThetaRecordParser


# class ThetaRecord(Record):
#     def __init__(self, string):
#         self.lexer = ThetaRecordLexer(string)
#
#     def _lexical_tokens(self):
#         pass
#
#     def ordered_pairs(self):
#         pass
#
#     def __str__(self):
#         result = ""
#         for token in self.lexer.tokens:
#             result += token.content
#         return super().__str__() + result
#
#
# model.get_records('THETA') =
#     0 : $THETA  (0,0.00469307) ; CL
#
#     1 : $THETA  (0,1.00916) ; V
#
#     2 : $THETA  (-.99,.1)
#
#
# model.get_records('THETA')[0].lexer =
# 0  ThetaRecordToken.WHITESPACE '  '
# 2  ThetaRecordToken.OPENPAREN  '('
# 3  ThetaRecordToken.TOKEN      '0'
# 4  ThetaRecordToken.COMMA      ','
# 5  ThetaRecordToken.TOKEN      '0.00469307'
# 15 ThetaRecordToken.CLOSEPAREN ')'
# 16 ThetaRecordToken.WHITESPACE ' '
# 17 ThetaRecordToken.COMMENT    '; CL'
# 21 ThetaRecordToken.WHITESPACE '\n'


class ThetaRecord(Record):
    def __init__(self, buf):
        self.parser = ThetaRecordParser(buf)
        self.root = self.parser.root
        # self.thetas = self.parser.root.all('theta')

    def _lexical_tokens(self):
        pass

    def ordered_pairs(self):
        pass

    def __str__(self):
        return super().__str__() + str(self.parser.root)
