import re

from pharmpy.model import ModelSyntaxError

from .records.factory import create_record
from .records.raw_record import RawRecord


class Abbreviated:
    """Handling all $ABBREVIATED in a control stream"""

    def __init__(self, stream):
        self.stream = stream

    @property
    def replace(self):
        """Get all $ABBR REPLACE as a dictionary"""
        d = dict()
        for record in self.stream.get_records('ABBREVIATED'):
            d.update(record.replace)
        return d


class NMTranParser:
    """Parser for NMTran control streams"""

    def parse(self, text):
        stream = NMTranControlStream()

        record_strings = re.split(r'^([ \t]*\$)', text, flags=re.MULTILINE)
        first = record_strings.pop(0)  # Empty if nothing before first record
        if first:
            stream.records.append(RawRecord(first))

        for separator, s in zip(record_strings[0::2], record_strings[1::2]):
            record = create_record(separator + s)
            stream.records.append(record)

        return stream


default_record_order = [
    'SIZES',
    'SUBROUTINES',
    'MODEL',
    'ABBREVIATED',
    'PK',
    'PRED',
    'DES',
    'ERROR',
    'THETA',
    'OMEGA',
    'SIGMA',
]


class NMTranControlStream:
    """Representation of a parsed control stream (model file)"""

    def __init__(self):
        self.records = []
        self._active_problem = 0
        self.abbreviated = Abbreviated(self)

    def get_records(self, name):
        """Return a list of all records of a certain type in the current $PROBLEM"""
        current_problem = -1
        found = []
        for record in self.records:
            if record.name == 'PROBLEM':
                current_problem += 1
            if current_problem == self._active_problem and record.name == name:
                found.append(record)
        return found

    def append_record(self, content):
        """Create and append record at the end"""
        record = create_record(content)
        self.records.append(record)
        return record

    def insert_record(self, content, at_index=None):
        """Create and insert a new record at the correct position

        If the record type is already present the new record will be put
        directly after the last record of that type.
        If no record of the type is present the new record will be put
        given the default record order.
        If record type is unknown. Place at the end.
        """

        record = create_record(content)
        name = record.name

        if at_index:
            self.records.insert(at_index, record)
            return record

        current_problem = -1
        index = None
        for i, currec in enumerate(self.records):
            if currec.name == 'PROBLEM':
                current_problem += 1
            if current_problem == self._active_problem and currec.name == name:
                index = i

        if index is None:
            try:
                default_order_index = default_record_order.index(name)
                before_records = default_record_order[0:default_order_index]
            except ValueError:
                before_records = []
            current_problem = -1
            for i, currec in enumerate(self.records):
                if currec.name == 'PROBLEM':
                    current_problem += 1
                if current_problem == self._active_problem and currec.name in before_records:
                    index = i

        if index is None:
            index = len(self.records)

        self.records.insert(index + 1, record)
        return record

    def remove_records(self, records):
        for rec in records:
            self.records.remove(rec)

    def go_through_omega_rec(self):
        next_omega = 1
        previous_size = None
        prev_cov = None

        for omega_record in self.get_records('OMEGA'):
            next_omega_cur = next_omega
            omegas, next_omega, previous_size = omega_record.parameters(
                next_omega_cur, previous_size
            )
            etas, next_eta, prev_cov, _ = omega_record.random_variables(next_omega_cur, prev_cov)

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
