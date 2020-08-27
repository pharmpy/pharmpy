import re

from pharmpy.model import ModelSyntaxError

from .records.factory import create_record
from .records.raw_record import RawRecord


class NMTranParser:
    """Parser for NMTran control streams
    """
    def parse(self, text):
        stream = NMTranControlStream()

        record_strings = re.split(r'^([ \t]*\$)', text, flags=re.MULTILINE)
        first = record_strings.pop(0)       # Empty if nothing before first record
        if first:
            stream.records.append(RawRecord(first))

        for separator, s in zip(record_strings[0::2], record_strings[1::2]):
            record = create_record(separator + s)
            stream.records.append(record)

        return stream


class NMTranControlStream:
    """Representation of a parsed control stream (model file)
    """
    def __init__(self):
        self.records = []
        self._active_problem = 0

    def get_records(self, name):
        """Return a list of all records of a certain type in the current $PROBLEM
        """
        current_problem = -1
        found = []
        for record in self.records:
            if record.name == 'PROBLEM':
                current_problem += 1
            if current_problem == self._active_problem and record.name == name:
                found.append(record)
        return found

    def append_record(self, content):
        """ Create and append record at the end
        """
        record = create_record(content)
        self.records.append(record)
        return record

    def insert_record(self, content, rec_type):
        """Create and insert a new record before a record with given type
        """
        current_problem = -1
        index = 0

        for record in self.records:
            if record.name == 'PROBLEM':
                current_problem += 1
            if current_problem == self._active_problem and record.name == rec_type:
                index = self.records.index(record)

        record = create_record(content)
        self.records.insert(index + 1, record)

        return record

    def remove_records(self, records):
        for rec in records:
            self.records.remove(rec)

    def validate(self):
        in_problem = False
        for record in self.records:
            if in_problem and record.name == 'SIZES':
                raise ModelSyntaxError('The SIZES record must come before the first PROBLEM record')
            elif record.name == 'PROBLEM':
                in_problem = True
            if hasattr(record, 'validate'):
                record.validate()

    def __str__(self):
        return ''.join(str(x) for x in self.records)
