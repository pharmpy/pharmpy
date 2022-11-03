import re
from itertools import chain

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
        return dict(
            chain.from_iterable(
                record.replace.items() for record in self.stream.get_records('ABBREVIATED')
            )
        )

    def translate_to_pharmpy_names(self):
        return dict(
            chain.from_iterable(
                record.translate_to_pharmpy_names().items()
                for record in self.stream.get_records('ABBREVIATED')
            )
        )


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

        in_problem = False
        for record in stream.records:
            if in_problem and record.name == 'SIZES':
                raise ModelSyntaxError('The SIZES record must come before the first PROBLEM record')
            elif record.name == 'PROBLEM':
                in_problem = True

        return stream


default_record_order = [
    'SIZES',
    'INPUT',
    'DATA',
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
    'ESTIMATION',
    'COVARIANCE',
    'TABLE',
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

    def _get_first_record(self, name):
        return next(iter(self.get_records(name)), None)

    def append_record(self, content):
        """Create and append record at the end"""
        record = create_record(content)
        self.records.append(record)
        return record

    def insert_record(self, content: str, at_index=None):
        """Create and insert a new record at the correct position

        If the record type is already present the new record will be put
        directly after the last record of that type.
        If no record of the type is present the new record will be put
        given the default record order.
        If record type is unknown. Place at the end.
        """

        record = create_record(content)
        name = record.name
        assert isinstance(name, str)

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

    def replace_records(self, old, new):
        keep = []
        first = True
        for rec in self.records:
            if rec not in old:
                keep.append(rec)
            else:
                if first:
                    keep.extend(new)
                    first = False
        self.records = keep

    def get_pred_pk_record(self):
        pred = self._get_first_record('PRED')
        if pred is not None:
            return pred

        pk = self._get_first_record('PK')
        if pk is not None:
            return pk

        raise ModelSyntaxError('Model has no $PK or $PRED')

    def get_pk_record(self):
        return self._get_first_record('PK')

    def get_error_record(self):
        return self._get_first_record('ERROR')

    def get_des_record(self):
        return self._get_first_record('DES')

    def __str__(self):
        return ''.join(map(str, self.records))
