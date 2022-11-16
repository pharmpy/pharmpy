"""
NONMEM abbreviated record class.
"""

import re
from itertools import islice

from .record import Record


def _is_quoted_with(q: str, s: str):
    return s.startswith(q) and s.endswith(q)


def _is_quoted(s: str):
    return _is_quoted_with("'", s) or _is_quoted_with('"', s)


def strip_quote(s):
    return s[1:-1] if _is_quoted(s) else s


class AbbreviatedRecord(Record):
    @property
    def replace(self):
        """Give a dict of all REPLACE in record"""
        d = {}
        for replace in self.root.subtrees('replace'):
            a, b = islice(iter(replace.leaves('ANY')), 2)
            first = strip_quote(a.value)
            second = strip_quote(b.value)
            d[first] = second
        return d

    def translate_to_pharmpy_names(self):
        return {value: re.sub(r'\((\w+)\)', r'_\1', key) for key, value in self.replace.items()}
