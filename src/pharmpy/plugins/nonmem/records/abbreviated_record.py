"""
NONMEM abbreviated record class.
"""

import re

from .record import Record


def strip_quote(s):
    if s.startswith("'") and s.endswith("'") or s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    else:
        return s


class AbbreviatedRecord(Record):
    @property
    def replace(self):
        """Give a dict of all REPLACE in record"""
        d = {}
        for replace in self.root.all('replace'):
            strings = replace.all('ANY')
            first = strip_quote(str(strings[0]))
            second = strip_quote(str(strings[1]))
            d[first] = second
        return d

    def translate_to_pharmpy_names(self):
        return {value: re.sub(r'\((\w+)\)', r'_\1', key) for key, value in self.replace.items()}
