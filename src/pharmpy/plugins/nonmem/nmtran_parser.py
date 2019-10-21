import re

from .records.raw_record import RawRecord
from .records.factory import create_record


class NMTranParser:
    """Parser for NMTran control streams
    """
    def parse(self, text):
        stream = NMTranControlStream()

        record_strings = re.split(r'^(?=\s*\$)', text, flags=re.MULTILINE)
        first = record_strings.pop(0)       # Empty if nothing before first record
        if first:
            stream.records.append(RawRecord(record_string.pop(0)))

        for s in record_strings:
            stream.records.append(create_record(s))

        return stream


class NMTranControlStream:
    """Representation of a parsed control stream (model file)
    """
    def __init__(self):
       self.records = []

    def __str__(self):
        return ''.join(str(x) for x in self.records)
