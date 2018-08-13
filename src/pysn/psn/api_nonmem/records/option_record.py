# -*- encoding: utf-8 -*-

"""
Generic NONMEM option record class.

Assumes 'KEY=VALUE' and doeos not support 'KEY VALUE'.
"""

import re
from collections import OrderedDict
from enum import Enum

from .record import Record


class OptionTokenType(Enum):
    TOKEN = 1
    WHITESPACE = 2
    CONTINUATION = 3        # &\n
    COMMENT = 4


class OptionToken:
    def __init__(self, t, content):
        self.type = t
        self.content = content


class OptionRecord(Record):
    def __init__(self, string):
        self.tokens = []
        while string:
            m = re.match(r'&\n', string)
            if m:
                token = OptionToken(OptionTokenType.CONTINUATION, m.group())
                self.tokens.append(token)
                string = string[m.end():]
                continue
            m = re.match(r'\s+', string)
            if m:
                token = OptionToken(OptionTokenType.WHITESPACE, m.group())
                self.tokens.append(token)
                string = string[m.end():]
                continue
            m = re.match(r';.*', string)
            if m:
                token = OptionToken(OptionTokenType.COMMENT, m.group())
                self.tokens.append(token)
                string = string[m.end():]
                continue
            m = re.match(r'\S+', string)        # FIXME: Den här suger nog upp för mycket
            token = OptionToken(OptionTokenType.TOKEN, m.group())
            self.tokens.append(token)
            string = string[m.end():]

    def __str__(self):
        result = ""
        for token in self.tokens:
            result += token.content
        return super().__str__() + result

    def _lexical_tokens(self):
        """
        Returns the lexical tokens.

        Ordered dictionary from lexical token to list of actual tokens. Will
            also merge continuation lines

        Example:
            'MAX&\nEVALS=0' gives {
                'MAXEVALS=0' : [
                    OptionToken(TOKEN, "MAX"),
                    OptionToken(CONTINUATION, "&\n"),
                    OptionToken(TOKEN, "EVALS=0)
                ]
            }
        """
        numtokens = len(self.tokens)
        i = 0
        current_token = ""
        lexical = OrderedDict()
        # lexical = []
        current_actual = []
        while i < numtokens:
            if self.tokens[i].type == OptionTokenType.TOKEN:
                current_token += self.tokens[i].content
                current_actual.append(self.tokens[i])
                if (i + 2 < numtokens and self.tokens[i + 1].type ==
                        OptionTokenType.CONTINUATION and self.tokens[i + 2].type
                        == OptionTokenType.TOKEN):
                    i += 2
                    continue
                lexical[current_token] = current_actual
            current_token = ""
            current_actual = []
            i += 1
        return lexical

    def ordered_pairs(self):
        """Returns all key value pairs as an ordered dictionary"""
        d = OrderedDict()
        for token in self._lexical_tokens():
            s = token.split('=')
            if len(s) > 1:
                d[s[0]] = s[1]
            else:
                d[s[0]] = None
        return d

    def set(self, name, value=None):
        if value:
            string = name + '=' + value
        else:
            string = name
        for tokstr, tokens in self._lexical_tokens().iteritems():
            a = tokstr.split('=')
            if a[0] == name:
                tokens[0].content = string
                for tok in tokens[1:]:      # If only one?
                    self.tokens.remove(tok)

    def delete(self, name):
        """Deletes the first option with name 'name'"""
        for tokstr, tokens in self._lexical_tokens().iteritems():
            a = tokstr.split('=')   # More elegant?
            if a[0] == name:        # Need abbrev match here
                for tok in tokens:
                    self.tokens.remove(tok)

    def get_value(self, name):
        for token in self._lexical_tokens():
            s = token.split('=')
            if self.description.match(s[0], name):
                return s[1]
