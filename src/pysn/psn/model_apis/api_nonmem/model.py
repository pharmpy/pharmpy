"""(Specific) NONMEM 7.x model API"""
from .records.factory import create_record
from ...exceptions import *
from .input import Input


def create_unique_symbol(symbols, prefix):
    """Create a unique symbol with prefix given a list of used symbols"""
    count = 1
    while True:
        candidate = prefix + str(count)
        if candidate in symbols:
            i += 1
        else:
            return candidate


class Model:
    """A NONMEM 7.x model"""
    def __init__(self, filename):
        """Loads model from file"""
        with open(filename, 'r') as model_file:
            content = model_file.read()

        record_strings = content.split('$')
        # The first comment does not belong to any record
        if not record_strings[0].startswith('$'):
            self.first_comment = record_strings[0]
            del record_strings[0]

        self.records = []
        for string in record_strings:
            new_record = create_record(string)
            self.records.append(new_record)

        self._validate()

        self.input = Input(self)

    def __str__(self):
        string = self.first_comment
        for record in self.records:
            string += str(record)
        return string

    def _validate(self):
        """Validates model syntactically"""
        # SIZES can only be first
        passed_sizes = False
        for record in self.records:
            if record.name == "SIZES":
                if passed_sizes:
                    raise ModelParsingError("The SIZES record must come before the first PROBLEM record")
            else:
                passed_sizes = True

    def get_records(self, name, problem=0):
        """ Get all records with a certain name for a certain problem"""
        result = []
        curprob = -1
        for record in self.records:
            if record.name == "PROBLEM":
                curprob += 1
            elif curprob == problem and record.name == name:
                result.append(record)
        return result
