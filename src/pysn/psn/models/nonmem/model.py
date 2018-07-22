from .records.factory import create_record
from ...model import ModelParsingError


def create_unique_symbol(symbols, prefix):
    """ Create a unique symbol with prefix given a list of used symbols
    """
    count = 1
    while True:
        candidate = prefix + str(count)
        if candidate in symbols:
            i += 1
        else:
            return candidate



class Input:
    def __init__(self, model):
        self.model = model

    def column_names(self, problem=0):
        """ Get a list of the column names of the input dataset
            Limitation: Tries to create unique symbol for anonymous columns, but only use the INPUT names
        """
        records = self.model.get_records("INPUT", problem=problem)
        all_symbols = []
        for record in records:
            pairs = record.ordered_pairs()
            for key in pairs:
                all_symbols.append(key)
                if pairs[key]:
                    all_symbols.append(key)
        names = []
        for record in records:
            pairs = record.ordered_pairs()
            for key, value in pairs.items():
                if key == "DROP" or key == "SKIP":
                    names.append(all_symbols, "DROP")
                else:
                    names.append(key)
        return names

    def dataset_filename(self, problem=0):
        """ Get the filename of the dataset
        """
        data_records = self.model.get_records("DATA", problem=problem)
        pairs = data_records[0].ordered_pairs()
        first_pair = next(iter(pairs.items()))
        return first_pair[0]


class Model:
    def __init__(self, filename):
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
        # SIZES can only be first
        passed_sizes = False
        for record in self.records:
            if record.name == "SIZES":
                if passed_sizes:
                    raise ModelParsingError("The SIZES record must come before the first PROBLEM record")
            else:
                passed_sizes = True

    def get_records(self, name, problem=0):
        """ Get all records with a certain name for a certain problem.
        """
        result = []
        curprob = -1
        for record in self.records:
            if record.name == "PROBLEM":
                curprob += 1
            elif curprob == problem and record.name == name:
                result.append(record)
        return result
