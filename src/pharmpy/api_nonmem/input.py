# -*- encoding: utf-8 -*-

import re
from io import StringIO
from os.path import realpath
from pathlib import Path

import pandas as pd

from pharmpy import input


class NMTRANDataIO(StringIO):
    """ An IO class that is a prefilter for pandas.read_table.
        Things that cannot be handled directly by pandas will be taken care of here and the
        rest will be taken care of by pandas.
        Currently it takes care of filtering out ignored rows
    """
    def __init__(self, filename_or_io, ignore_character):
        """ filename_or_io is a string with a path, a path object or any IO object, i.e. StringIO
        """
        if hasattr(filename_or_io, 'read'):
            contents = filename_or_io.read()
        else:
            with open(str(filename_or_io), 'r') as datafile:
                contents = datafile.read()      # All variations of newlines are converted into \n

        if ignore_character:
            if ignore_character == '@':
                comment_regexp = re.compile(r'^[A-Za-z].*\n', re.MULTILINE)
            else:
                comment_regexp = re.compile('^[' + ignore_character + '].*\n', re.MULTILINE)
            contents = re.sub(comment_regexp, '', contents)

        super().__init__(contents)


class ModelInput(input.ModelInput):
    """A NONMEM 7.x model input class. Covers at least $INPUT and $DATA."""

    @property
    def path(self):
        record = self.model.get_records('DATA')[0]
        path = Path(record.filename)
        if not path.is_absolute():
            path = self.model.path.parent / path
        try:
            return path.resolve()
        except FileNotFoundError:
            return path

    @path.setter
    def path(self, path):
        path = Path(path)
        assert not path.exists() or path.is_file(), ('input path change, but non-file exists at '
                                                     'target (%s)' % str(path))
        record = self.model.get_records('DATA')[0]
        # super().path = path
        self.logger.info('Setting %r.path to %r', repr(self), str(path))
        record.filename = str(path)

    def repath(self, relpath):
        """Re-calculate path with model location update.

        Caller is :class:`~pharmpy.source_io.SourceResource`. Source path change has changed, likely
        before initiating a filesystem write for a copy.

        Arguments:
            relpath: Maps new dir -> old (current) dir.

        :attr:`~ModelInput.path` is prefixed *relpath* if given. Otherwise, :attr:`~ModelInput.path`
        is made absolute to ensure it resolves from new path.

        .. note:: No change if :attr:`~ModelInput.path` is already absolute.
        """
        record = self.model.get_records('DATA')[0]
        path = Path(record.filename)
        if not path.is_absolute():
            if relpath:
                self.path = relpath / path
            else:
                try:
                    self.path = path.resolve()
                except FileNotFoundError:
                    self.path = Path(realpath(str(path)))

    @property
    def data_frame(self):
        try:
            return self._data_frame
        except AttributeError:
            self._read_data_frame()
        return self._data_frame

    def _column_names(self):
        input_records = self.model.get_records("INPUT")
        for record in input_records:
            for key, value in record.option_pairs.items():
                if value:
                    if key == 'DROP' or key == 'SKIP':
                        yield value
                    else:
                        yield key
                else:
                    yield key

    @staticmethod
    def _postprocess_data_frame(df, column_names, null_token=0):
        """ Do the following changes to the data_frame after reading it in
            1. Replace all NaN with the null_token
            2. Pad with null_token columns if $INPUT has more columns than the dataset 
            3. Set column names from $INPUT and pad with None dataset has more columns
        """
        df.fillna(null_token, inplace=True)
        colnames = list(column_names)
        coldiff = len(colnames) - len(df.columns)   # Difference between number of columns in $INPUT and in the dataset
        if coldiff > 0:
            for _ in range(coldiff):    # Create empty columns. Pandas does not support df[[None, None]] = [0, 0] or similar hence the loop
                df[None] = null_token
        elif coldiff < 0:
            colnames += [None] * abs(coldiff)
        df.columns = colnames

    def _read_data_frame(self):
        data_records = self.model.get_records('DATA')
        ignore_character = data_records[0].ignore_character
        self._data_frame = ModelInput.read_dataset(self.path, self._column_names(), ignore_character=ignore_character)

    @staticmethod
    def read_dataset(filename_or_io, colnames, ignore_character='@'):
        """ A static method to read in a NM-TRAN dataset and return a data frame
        """
        file_io = NMTRANDataIO(filename_or_io, ignore_character)
        df = pd.read_table(file_io, sep=r' *, *| *[\t] *| +', na_values='.', header=None, engine='python')
        ModelInput._postprocess_data_frame(df, colnames)
        return df

    @property
    def filters(self):
        data_records = self.model.get_records("DATA")
        return data_records[0].filters

    @filters.setter
    def filters(self, f):
        data_records = self.model.get_records("DATA")
        data_records[0].filters = f

    @property
    def id_column(self):
        colnames = self._column_names()
        if 'ID' in colnames:
            return 'ID'
        if 'L1' in colnames:
            return 'L1'
        raise KeyError('Dataset does not have an ID column')
