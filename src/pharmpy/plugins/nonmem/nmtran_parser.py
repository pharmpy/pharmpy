import re
from pathlib import Path
from lark import Lark

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
            record, content = create_record(s)
            if type(record) != RawRecord:
                self._parse_record(record, content)
            stream.records.append(record)

        return stream

    _grammar_root = Path(__file__).parent.resolve() / 'grammars'

    def _parse_record(self, record, content):
        class_name = type(record).__name__
        record_name = class_name[:-6].lower()
        grammar_filename = f'{record_name}_record.lark'
        path = NMTranParser._grammar_root / grammar_filename
        parser = Lark(open(path))
        parser.parse(content)


class NMTranControlStream:
    """Representation of a parsed control stream (model file)
    """
    def __init__(self):
       self.records = []

    def __str__(self):
        return ''.join(str(x) for x in self.records)
