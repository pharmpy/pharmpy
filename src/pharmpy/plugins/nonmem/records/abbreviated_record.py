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
        d = dict()
        for replace in self.root.all("replace"):
            strings = [str(node) for node in replace.all("ANY")]
            first = strip_quote(strings[0])
            second = strip_quote(strings[1])
            d[first] = second
        return d

    def translate_to_pharmpy_names(self):
        parameter_names = dict()
        for key, value in self.replace.items():
            key_new = re.sub(r'\((\w+)\)', r'_\1', key)
            parameter_names[value] = key_new
        return parameter_names
