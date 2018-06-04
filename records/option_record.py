from enum import Enum
import re

# Multilayered global options list in the records_list file??
# Only one class for optionrecord
# New objects can be created with factory method

#from records.record import Record
from records.option_description import OptionDescriptionList, OptionType
import records.record

class OptionTokenType(Enum):
    TOKEN = 1
    WHITESPACE = 2
    CONTINUATION = 3        # &\n
    COMMENT = 4

class OptionToken:
    def __init__(self, t, content):
        self.type = t
        self.content = content

class OptionRecord(records.record.Record):
    def __init__(self, string, description):
        self.description = description
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
            m = re.match(r'\S+', string)
            token = OptionToken(OptionTokenType.TOKEN, m.group())
            self.tokens.append(token)
            string = string[m.end():]

    def __str__(self):
        result = ""
        for token in self.tokens:
            result += token.content
        return result

    def lexical_tokens(self):
        """ Generates the lexical tokens. Will also merge continuation lines
        """
        numtokens = len(self.tokens)
        i = 0
        current_token = ""
        while i < numtokens:
            if self.tokens[i].type == OptionTokenType.TOKEN:
               current_token += self.tokens[i].content
            if i + 2 < numtokens and self.tokens[i + 1].type == OptionTokenType.CONTINUATION and \
                self.tokens[i + 2].type == OptionTokenType.TOKEN:
                i += 2
                continue
            yield current_token
            current_token = ""
            i += 1

    # Generator for all key value pairs
    # Need heuristics for MAX 9999. As 9999 is number connected to MAX
    def pairs(self):
        for token in self.tokens:
            if token.type == OptionTokenType.TOKEN:
                s = token.content.split('=')
                if len(s) > 1:
                    yield (s[0], s[1])
                else:
                    yield (s[0], None)

    def set(self, name, value=None):
        if value:
            string = name + '=' + value
        else:
            string = name
        for i, token in enumerate(self.tokens):         # Use generator instead
            if token.type == OptionTokenType.TOKEN:
                final_token_index = i
                s = token.content.split('=')
                if self.description.match(s[0], name):
                    if value:                   # Is this really correct?
                        token.content = string
                    return
        self.tokens.insert(final_token_index + 1, OptionToken(OptionTokenType.TOKEN, string))
        self.tokens.insert(final_token_index + 1, OptionToken(OptionTokenType.WHITESPACE, " "))

    def delete(self, name):
        remove = []
        for token in self.tokens:
            if token.type == OptionTokenType.TOKEN:
                s = token.content.split('=')
                if self.description.match(s[0], name):
                    remove.append(token)
        for token in remove:
            self.tokens.remove(token)

    def get_value(self, name):
        for token in self.tokens:
            if token.type == OptionTokenType.TOKEN:
                s = token.content.split('=')
                if self.description.match(s[0], name):
                    return s[1]

class SizesRecord(OptionRecord):
    def __init__(self, content):
        structure = [
            { 'name' : 'LTH', 'type' : OptionType.VALUE, 'abbreviate' : False },
        ]
        super().__init__(content, OptionDescriptionList(structure))
