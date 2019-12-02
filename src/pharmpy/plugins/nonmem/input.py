# -*- encoding: utf-8 -*-

import re
import warnings
from io import StringIO
from os.path import realpath
from pathlib import Path

import numpy as np
import pandas as pd

from pharmpy import input
from pharmpy.input import DatasetError


def convert_fortran_exp_format(number_string):
    """This function will try to convert the number_string from the special fortran exponential format
       into an np.float64. It covers "1d1", "1D1", "a+b", "a-b", "+" and "-". All other cases will return None to
       signal that the number_string is not of the special form.
    """
    #FIXME: Move this function as it will also be used by the output parser (and is perhaps more important there)
    if number_string == '+' or number_string == '-':
        return 0.0

    m = re.match(r'((\.)|(\.\d+)|(\d+\.)|(\d+\.\d+))([+-]\d+)', number_string)
    if m:
        if m.group(2):
            mantissa = 0.0
        elif m.group(3):
            mantissa = float(m.groups(3))
        elif m.group(4):
            mantissa = float(m.groups(4))
        elif m.group(5):
            mantissa = float(m.groups(5))
        exponent = m.group(6)
        return np.float64(mantissa * 10**exponent)

    if "D" in number_string or "d" in number_string:
        clean_number = number_string.replace("D", "e").replace("d", "e")
        try:
            y = np.float64(clean_number)
        except:
            return None
        return y

    return None


class NMTRANDataIO(StringIO):
    """ An IO class that is a prefilter for pandas.read_table.
        Things that must be done before using pandas will be done here.
        Currently it takes care of filtering out ignored rows and handles special delimiter cases
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

        if re.search(r' \t', contents):     # Space before TAB not allowed (see documentation)
            raise DatasetError("The dataset contains a TAB preceeded by a space, which is not allowed by NM-TRAN")

        #if re.search(r'^[ \t]*\n', re.MULTILINE):       # Blank lines
        #    raise DatasetError("The dataset contains one or more blank lines. This is not allowed by NM-TRAN without the BLANKOK option")

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
        """List all column names in order.
           Use the synonym in case of synonyms. Empty string in case of only DROP or SKIP.
        """
        _reserved_column_names = ['ID', 'L1', 'L2', 'DV', 'MDV', 'RAW_', 'MRG_', 'RPT_',
            'TIME', 'DATE', 'DAT1', 'DAT2', 'DAT3', 'EVID', 'AMT', 'RATE', 'SS', 'II', 'ADDL',
            'CMT', 'PCMT', 'CALL', 'CONT' ]

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
    def _convert_data_item(x, null_value):
        if x is None or x == '.' or x == '':
            x = null_value
        if len(x) > 24:
            raise DatasetError("The dataset contains an item that is longer than 24 characters")
        try:
            y = np.float64(x)
        except:
            y = convert_fortran_exp_format(x)
            if y is None:
                raise DatasetError("The dataset contains an invalid number")
        return y

    @staticmethod
    def _postprocess_data_frame(df, column_names, null_value):
        """ Do the following changes to the data_frame after reading it in
            1. Convert ordinary floating point numbers to float64
            2. Convert numbers of special fortran format to float64
            3. Convert None, '.', empty string to the NULL value
            4. Convert Inf/NaN properly
            5. Pad with null_token columns if $INPUT has more columns than the dataset 
            6. Stip away superfluous columns from the dataset and give a warning
        """
        for column in df:
            df[column] = df[column].apply(ModelInput._convert_data_item, args=(null_value,))

        colnames = list(column_names)
        coldiff = len(colnames) - len(df.columns)   # Difference between number of columns in $INPUT and in the dataset
        if coldiff > 0:
            for _ in range(coldiff):    # Create empty columns. Pandas does not support df[[None, None]] = [0, 0] or similar hence the loop
                df[None] = float(null_value)
        elif coldiff < 0:
            warnings.warn(DatasetWarning("There are more columns in the dataset than in $INPUT. The extra columns have not been loaded."))
        df.columns = colnames

    def _read_data_frame(self):
        data_records = self.model.get_records('DATA')
        ignore_character = data_records[0].ignore_character
        null_value = data_records[0].null_value
        self._data_frame = ModelInput.read_dataset(self.path, self._column_names(), ignore_character=ignore_character, null_value=null_value)

    @staticmethod
    def read_dataset(filename_or_io, colnames, ignore_character='@', null_value='0'):
        """ A static method to read in an NM-TRAN dataset and return a data frame
        """
        file_io = NMTRANDataIO(filename_or_io, ignore_character)
        df = pd.read_table(file_io, sep=r' *, *| *[\t] *| +', na_filter=False, header=None, engine='python', quoting=3, dtype=np.object)
        ModelInput._postprocess_data_frame(df, colnames, str(null_value))
        return df

    @property
    def id_column(self):
        colnames = self._column_names()
        if 'ID' in colnames:
            return 'ID'
        if 'L1' in colnames:
            return 'L1'
        raise KeyError('Dataset does not have an ID column')
