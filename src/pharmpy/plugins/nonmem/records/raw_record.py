from dataclasses import dataclass

from .record import Record


@dataclass(frozen=True)
class RawRecord(Record):
    """A record that just keeps contents unparsed.
    Used for unknown records and for anything coming before the first record
    """

    buffer: str

    @property
    def tree(self):
        raise TypeError('Cannot access root of raw record')
