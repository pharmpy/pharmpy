# -*- encoding: utf-8 -*-

from . import record


class RawRecord(record.Record):
    """A record that just keeps contents unparsed.
    Used for unknown records and for anything coming before the first record
    """

    def __init__(self, content):
        self.content = content
        self.raw_name = ''

    def __str__(self):
        return self.raw_name + self.content
