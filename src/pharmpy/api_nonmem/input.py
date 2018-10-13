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
    """
    def __init__(self, filename, ignore_character):
        with open(str(filename), 'r') as datafile:
            contents = datafile.read()      # All variations of newlines are converted into \n

        if ignore_character:
            if ignore_character == '@':
                comment_regexp = re.compile(r'^[A-Za-z].*\n', re.MULTILINE)
            else:
                comment_regexp = re.compile('^[' + ignore_character + '].*\n', re.MULTILINE)
            contents = re.sub(comment_regexp, '', contents)

        # Replace dot surrounded by space with 0 as explained in the NM-TRAN manual
        contents = re.sub(r'\s\.\s', '0', contents)

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

    def _read_data_frame(self):
        data_records = self.model.get_records("DATA")
        ignore_character = data_records[0].ignore_character
        file_io = NMTRANDataIO(self.path, ignore_character)
        self._data_frame = pd.read_table(file_io, sep='\s+|,', header=None, engine='python')
        self._data_frame.columns = list(self._column_names())

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
