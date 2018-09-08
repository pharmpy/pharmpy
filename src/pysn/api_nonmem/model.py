# -*- encoding: utf-8 -*-

"""(Specific) NONMEM 7.x model API."""

from pysn import generic
from .input import ModelInput
from .output import ModelOutput
from .parameters import ParameterModel
from .records.factory import create_record


def create_unique_symbol(symbols, prefix):
    """Creates a unique symbol with prefix given a list of used symbols"""
    count = 1
    while True:
        candidate = prefix + str(count)
        if candidate in symbols:
            count += 1
        else:
            return candidate


class Model(generic.Model):
    """A NONMEM 7.x model"""

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, new):
        pos = None
        prob_i = -1
        for i, record in enumerate(self.records):
            if record.name != 'PROBLEM':
                continue
            prob_i += 1
            if pos:
                pos = (pos[0], i)
                break
            elif prob_i == new or record.string == new:
                pos = (i, None)
        if not pos:
            raise generic.ModelLookupError(new)
        elif not pos[1]:
            pos = (pos[0], i)
        self._index = prob_i
        self._index_records = pos

    def load(self):
        record_strings = self.content.split('$')
        # The first comment does not belong to any record
        if not record_strings[0].startswith('$'):
            self.first_comment = record_strings[0]
            del record_strings[0]

        self.records = []
        for string in record_strings:
            new_record = create_record(string)
            self.records.append(new_record)

        self.index = 0
        self.input = ModelInput(self)
        self.output = ModelOutput(self)
        self.parameters = ParameterModel(self)
        self.validate()

    def validate(self):
        """Validates model syntactically"""
        # SIZES can only be first
        assert self._index == 0
        for i, record in enumerate(self.records):
            if i < self._index_records[0]:
                continue
            if record.name == 'SIZES':
                raise generic.ModelParsingError(
                    'The SIZES record must come before the first PROBLEM record'
                )

    def get_records(self, name):
        """Get all records with a certain name"""
        result = []
        pos = self._index_records
        for record in self.records[pos[0]:pos[1]]:
            if record.name == name:
                result.append(record)
        return result

    def __str__(self):
        string = self.first_comment
        for record in self.records:
            string += str(record)
        return string
