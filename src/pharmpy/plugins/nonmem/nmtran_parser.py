import re
from pathlib import Path
from lark import Lark

from .records.raw_record import RawRecord
from .records.factory import create_record
from pharmpy.parse_utils.generic import AttrTree


class NMTranParser:
    """Parser for NMTran control streams
    """
    def parse(self, text):
        stream = NMTranControlStream()

        record_strings = re.split(r'^(?=[ \t]*\$)', text, flags=re.MULTILINE)
        first = record_strings.pop(0)       # Empty if nothing before first record
        if first:
            stream.records.append(RawRecord(first))

        for s in record_strings:
            record = create_record(s)
            stream.records.append(record)

        return stream

 

class NMTranControlStream:
    """Representation of a parsed control stream (model file)
    """
    def __init__(self):
       self.records = []

    def __str__(self):
        return ''.join(str(x) for x in self.records)
