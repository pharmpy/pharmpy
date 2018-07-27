import re
from enum import Enum
from collections import OrderedDict

from .record import Record

# Forms:
# 1. init [FIXED]
# 2. ([low,] init [,up] [FIXED]) 
# 3. ([low,] init [,up]) [FIXED]
# 4. (low,,up)
# 5. (value)xn 

class ThetaTokenType(Enum):
    OPENPAREN = 1
    CLOSEPAREN = 2
    COMMA = 3
    WHITESPACE = 4
    COMMENT = 5
    CONTINUATION = 6
    TOKEN = 7

class ThetaToken:
    def __init__(self, t, content):
        self.type = t
        self.content = content

class ThetaRecord(Record):
    def __init__(self, string):
        self.tokens = []
        while string:
            m = re.match(r'&\n', string)
            if m:
                token = ThetaToken(ThetaTokenType.CONTINUATION, m.group())
                self.tokens.append(token)
                string = string[m.end():]
                continue
            m = re.match(r'\s+', string)
            if m:
                token = ThetaToken(ThetaTokenType.WHITESPACE, m.group())
                self.tokens.append(token)
                string = string[m.end():]
                continue
            m = re.match(r';.*', string)
            if m:
                token = ThetaToken(ThetaTokenType.COMMENT, m.group())
                self.tokens.append(token)
                string = string[m.end():]
                continue
            m = re.match(r'\(', string)
            if m:
                token = ThetaToken(ThetaTokenType.OPENPAREN, m.group())
                self.tokens.append(token)
                string = string[m.end():]
                continue
            m = re.match(r'\)', string)
            if m:
                token = ThetaToken(ThetaTokenType.CLOSEPAREN, m.group())
                self.tokens.append(token)
                string = string[m.end():]
                continue
            m = re.match(r',', string)
            if m:
                token = ThetaToken(ThetaTokenType.COMMA, m.group())
                self.tokens.append(token)
                string = string[m.end():]
                continue
            m = re.match(r'[A-Za-z0-9=.+-]+', string)
            token = ThetaToken(ThetaTokenType.TOKEN, m.group())
            self.tokens.append(token)
            string = string[m.end():]

    def __str__(self):
        result = ""
        for token in self.tokens:
            result += token.content
        return super().__str__() + result
