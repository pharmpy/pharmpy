from .records.factory import create_record


class Input:
    """A NONMEM 7.x model $INPUT class"""
    def __init__(self, model):
        self.model = model

    def column_names(self, problem=0):
        """Gets a list of the column names of the input dataset

        Limitation: Tries to create unique symbol for anonymous columns, but
        only uses the INPUT names.
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
        """Get the filename of the dataset"""
        data_records = self.model.get_records("DATA", problem=problem)
        pairs = data_records[0].ordered_pairs()
        first_pair = next(iter(pairs.items()))
        return first_pair[0]
