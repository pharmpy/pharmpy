"""
NONMEM abbreviated record class.
"""

import re
from dataclasses import dataclass
from functools import cached_property
from itertools import islice

from pharmpy.internals.immutable import frozenmapping

from .record import ReplaceableRecord, with_parsed_and_generated


def _is_quoted_with(q: str, s: str):
    return s.startswith(q) and s.endswith(q)


def _is_quoted(s: str):
    return _is_quoted_with("'", s) or _is_quoted_with('"', s)


def _strip_quote(s: str):
    return s[1:-1] if _is_quoted(s) else s


def _strip_quote_token(t):
    return _strip_quote(t.value)


@with_parsed_and_generated
@dataclass(frozen=True)
class AbbreviatedRecord(ReplaceableRecord):
    @cached_property
    def replace(self):
        """Give a dict of all REPLACE in record"""
        return frozenmapping(
            tuple(islice(map(_strip_quote_token, iter(replace.leaves('ANY'))), 2))
            for replace in self.tree.subtrees('replace')
        )

    @cached_property
    def translate_to_pharmpy_names(self):
        return frozenmapping(
            (value, re.sub(r'\((\w+)\)', r'_\1', key)) for key, value in self.replace.items()
        )
