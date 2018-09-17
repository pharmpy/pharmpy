# -*- encoding: utf-8 -*-

"""(Specific) NONMEM 7.x model API."""

from pharmpy import generic
from pharmpy.generic import ModelParsingError

from .execute import NONMEM7
from .input import ModelInput
from .output import ModelOutput
from .parameters import ParameterModel
from .source_io import SourceResource


def create_unique_symbol(symbols, prefix):
    """Creates a unique symbol with prefix given a list of used symbols"""
    count = 1
    while True:
        candidate = prefix + str(count)
        if candidate in symbols:
            count += 1
        else:
            return candidate


def validate_records(records):
    """Validates NONMEM model (records) syntactically."""
    in_problem = False
    for record in records:
        if in_problem and record.name == 'SIZES':
            raise ModelParsingError('The SIZES record must come before the first PROBLEM record')
        if record.name == 'PROBLEM':
            in_problem = True


class Model(generic.Model):
    """A NONMEM 7.x model"""

    SourceResource = SourceResource
    Engine = NONMEM7
    ModelInput = ModelInput
    ModelOutput = ModelOutput
    ParameterModel = ParameterModel

    @property
    def records(self):
        records = list(self.source.input.iter_records(self.index))
        validate_records(records)
        return records

    def get_records(self, name):
        """Returns all records of a certain type."""
        return list(filter(lambda x: x.name == name, self.records))

    def __str__(self):
        return ''.join(str(x) for x in self.records)
