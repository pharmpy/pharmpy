# -*- encoding: utf-8 -*-

from . import record


class RawRecord(record.Record):
    """A record that just keeps contents unparsed."""
    def __init__(self, content):
        self.content = content

    def __str__(self):
        return super().__str__() + self.content
