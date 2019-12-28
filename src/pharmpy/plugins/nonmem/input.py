from os.path import realpath
from pathlib import Path


from pharmpy import input
import pharmpy.data
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
        #self.logger.info('Setting %r.path to %r', repr(self), str(path))
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
        _reserved_column_names = ['ID', 'L1', 'L2', 'DV', 'MDV', 'RAW_', 'MRG_', 'RPT_',
            'TIME', 'DATE', 'DAT1', 'DAT2', 'DAT3', 'EVID', 'AMT', 'RATE', 'SS', 'II', 'ADDL',
            'CMT', 'PCMT', 'CALL', 'CONT' ]
        if key in _reserved_column_names:
            return (key, value)
        elif value in _reserved_column_names:
            return (value, key)
        else:
            raise DatasetError(f'A column name "{key}" in $INPUT has a synonym to a non-reserved column name "{value}"')

    def _column_info(self):
        """List all column names in order.
           return tuple of three lists, colnames, coltypes and drop
           FIXME: Use the synonym in case of synonyms. Empty string in case of only DROP or SKIP.
        """

        input_records = self.model.control_stream.get_records("INPUT")
        colnames = []
        coltypes = []
        drop = []
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
                        (reserved_name, synonym) = ModelInput._synonym()
                        colnames.append(synonym)
                else:
                    if key == 'DROP' or key == 'SKIP':
                        drop.append(True)
                    else:
                        drop.append(False)
                    colnames.append(key)
                    reserved_name = key
                coltypes.append(pharmpy.data.read.infer_column_type(reserved_name)) 
        return colnames, coltypes, drop


    def _read_dataset(self, raw=False, parse_columns=tuple()):
        data_records = self.model.control_stream.get_records('DATA')
        ignore_character = data_records[0].ignore_character
        null_value = data_records[0].null_value
        (colnames, coltypes, drop) = self._column_info()
        if raw:
            ignore = None
            accept = None
        else:
            ignore = data_records[0].ignore
            accept = data_records[0].accept
        df = pharmpy.data.read_nonmem_dataset(self.path, raw, ignore_character, colnames, coltypes, drop,
                 null_value=null_value, parse_columns=parse_columns, ignore=ignore, accept=accept)
        df.name = self.path.stem
        return df
