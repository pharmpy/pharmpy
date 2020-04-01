from pathlib import Path

import pharmpy.data
from pharmpy import input
from pharmpy.data import DatasetError


class ModelInput(input.ModelInput):
    """A NONMEM 7.x model input class. Covers at least $INPUT and $DATA."""

    def __init__(self, model):
        self._dataset_updated = False
        super().__init__(model)

    @property
    def path(self):
        # FIXME: Should this really be public? Will follow source model
        record = self.model.control_stream.get_records('DATA')[0]
        path = Path(record.filename)
        if not path.is_absolute():
            path = self.model.source.path.parent / path     # Relative model source file.
        try:
            return path.resolve()
        except FileNotFoundError:
            return path

    @path.setter
    def path(self, path):
        path = Path(path)
        assert not path.exists() or path.is_file(), ('input path change, but non-file exists at '
                                                     'target (%s)' % str(path))
        record = self.model.control_stream.get_records('DATA')[0]
        record.filename = str(path)

    @property
    def dataset(self):
        try:
            return self._data_frame
        except AttributeError:
            self._data_frame = self._read_dataset(raw=False)
        return self._data_frame

    @dataset.setter
    def dataset(self, df):
        self._dataset_updated = True
        self._data_frame = df

    def read_raw_dataset(self, parse_columns=tuple()):
        return self._read_dataset(raw=True, parse_columns=parse_columns)

    @staticmethod
    def _synonym(key, value):
        """Return a tuple reserved name and synonym
        """
        _reserved_column_names = [
            'ID', 'L1', 'L2', 'DV', 'MDV', 'RAW_', 'MRG_', 'RPT_',
            'TIME', 'DATE', 'DAT1', 'DAT2', 'DAT3', 'EVID', 'AMT', 'RATE', 'SS', 'II', 'ADDL',
            'CMT', 'PCMT', 'CALL', 'CONT'
        ]
        if key in _reserved_column_names:
            return (key, value)
        elif value in _reserved_column_names:
            return (value, key)
        else:
            raise DatasetError(f'A column name "{key}" in $INPUT has a synonym to a non-reserved '
                               f'column name "{value}"')

    def _column_info(self):
        """List all column names in order.
            Use the synonym when synonym exists.
            return tuple of three lists, colnames, coltypes and drop together with a dictionary
            of replacements for reserved names (aka synonyms).
        """
        input_records = self.model.control_stream.get_records("INPUT")
        colnames = []
        coltypes = []
        drop = []
        synonym_replacement = {}
        for record in input_records:
            for key, value in record.option_pairs.items():
                if value:
                    if key == 'DROP' or key == 'SKIP':
                        drop.append(True)
                        colnames.append(value)
                        reserved_name = value
                    elif value == 'DROP' or value == 'SKIP':
                        colnames.append(key)
                        drop.append(True)
                        reserved_name = key
                    else:
                        drop.append(False)
                        (reserved_name, synonym) = ModelInput._synonym(key, value)
                        synonym_replacement[reserved_name] = synonym
                        colnames.append(synonym)
                else:
                    if key == 'DROP' or key == 'SKIP':
                        drop.append(True)
                    else:
                        drop.append(False)
                    colnames.append(key)
                    reserved_name = key
                coltypes.append(pharmpy.data.read.infer_column_type(reserved_name))
        return colnames, coltypes, drop, synonym_replacement

    def _update_input(self, new_names):
        """Update $INPUT with new column names

           currently supporting append columns at end
        """
        colnames, _, _, _ = self._column_info()
        appended_names = new_names[len(colnames):]
        input_records = self.model.control_stream.get_records("INPUT")
        last_input_record = input_records[-1]
        for colname in appended_names:
            last_input_record.append_option(colname)

    def _replace_synonym_in_filters(filters, replacements):
        result = []
        for f in filters:
            if f.COLUMN in replacements:
                s = ''
                for child in f.children:
                    if child.rule == 'COLUMN':
                        value = replacements[f.COLUMN]
                    else:
                        value = str(child)
                    s += value
            else:
                s = str(f)
            result.append(s)
        return result

    def _read_dataset(self, raw=False, parse_columns=tuple()):
        data_records = self.model.control_stream.get_records('DATA')
        ignore_character = data_records[0].ignore_character
        null_value = data_records[0].null_value
        (colnames, coltypes, drop, replacements) = self._column_info()

        if raw:
            ignore = None
            accept = None
        else:
            # FIXME: All direct handling of control stream spanning
            # over one or more records should move
            ignore = data_records[0].ignore
            accept = data_records[0].accept
            # FIXME: This should really only be done if setting the dataset
            if ignore:
                ignore = ModelInput._replace_synonym_in_filters(ignore, replacements)
            else:
                accept = ModelInput._replace_synonym_in_filters(accept, replacements)

        df = pharmpy.data.read_nonmem_dataset(self.path, raw, ignore_character, colnames, coltypes,
                                              drop, null_value=null_value,
                                              parse_columns=parse_columns, ignore=ignore,
                                              accept=accept)
        df.name = self.path.stem
        return df
