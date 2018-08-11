import re

from .record import Record
from pysn.parse_utils.oldlexer import Lexer


class ThetaRecordLexer(Lexer):
    '''Lexer for THETA records

    Legal forms:
        1. init [FIXED]
        2. ([low,] init [,up] [FIXED])
        3. ([low,] init [,up]) [FIXED]
        4. (low,,up)
        5. (value)xn
    '''
    # debug = True
    root = 'ThetaRecordToken'
    rules = {
        'root': [
            ('&\n', 'CONTINUATION'),
            ('\s+', 'WHITESPACE'),
            (';.*', 'COMMENT'),
            ('\(', 'OPENPAREN'),
            ('\)', 'CLOSEPAREN'),
            (',', 'COMMA'),
            ('FIX(ED)?', 'FIXED'),
            ('[A-Za-z0-9=.+-]+', 'TOKEN'),
        ],
    }
