# Load a NONMEM dataset
import warnings
import re
from io import StringIO

import numpy as np
import pandas as pd

import pharmpy.data
from pharmpy.data import DatasetError
from pharmpy.data import DatasetWarning
from pharmpy.data import ColumnType


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


def convert_fortran_number(number_string):
    """This function will try to convert the number_string from the general fortran exponential format
       into an np.float64. It covers "1d1", "1D1", "a+b", "a-b", "+" and "-". All other cases will return None to
       signal that the number_string is not of the special form.
    """
    try:
        y = np.float64(number_string)
        return y
    except (TypeError, ValueError):
        pass

    if number_string == '+' or number_string == '-':
        return 0.0

    m = re.match(r'([+\-]?)([^+\-dD]*)([+-])([^+\-dD]*)', number_string)
    if m:
        mantissa_sign = '-' if m.group(1) == '-' else ''
        mantissa = m.group(2)
        exponent_sign = m.group(3) 
        exponent = m.group(4)
        return np.float64(mantissa_sign + mantissa + "E" + exponent_sign + exponent)

    if "D" in number_string or "d" in number_string:
        clean_number = number_string.replace("D", "e").replace("d", "e")
        try:
            y = np.float64(clean_number)
            return y
        except (TypeError, ValueError):
            pass

    raise ValueError(f"Could not convert the fortran number {number_string} to float")


def _convert_data_item(x, null_value):
    if x is None or x == '.' or x == '':
        x = null_value
    if len(x) > 24:
        raise DatasetError("The dataset contains an item that is longer than 24 characters")
    return convert_fortran_number(x)


def infer_column_type(colname):
    """If possible infer the column type from the column name else use unknown
    """
    if colname == 'ID' or colname == 'L1':
        return ColumnType.ID
    else:
        return ColumnType.UNKNOWN


def _read_raw_dataset_from_file(filename_or_io, ignore_character):
    """Read a dataset without adding column names or column types 
    """
    file_io = NMTRANDataIO(filename_or_io, ignore_character)
    df = pd.read_table(file_io, sep=r' *, *| *[\t] *| +', na_filter=False, header=None, engine='python', quoting=3, dtype=np.object)
    return pharmpy.data.PharmDataFrame(df)


def _postprocess_data_frame(df, column_names, column_types, drop, null_value):
    """ Do the following changes to the data_frame after reading it in
        1. Convert ordinary floating point numbers to float64
        2. Convert numbers of special fortran format to float64
        3. Convert None, '.', empty string to the NULL value
        4. Convert Inf/NaN properly
        5. Pad with null_token columns if $INPUT has more columns than the dataset 
        6. Strip away superfluous columns from the dataset and give a warning
        # FIXME: Should remove DROPPed columns from dataset
    """
    coldiff = len(column_names) - len(df.columns)   # Difference between number of columns in $INPUT and in the dataset

    # Remove extra columns before parsing columns
    if coldiff < 0:
        warnings.warn(DatasetWarning("There are more columns in the dataset than in $INPUT. The extra columns have not been loaded."))
        #FIXME: Should we really warn here? Does NMTRAN care about errors in these columns?
        df = df[df.columns[0:coldiff]]

    # FIXME: Must be unique! No redefinition allowed in NONMEM. Where to check for duplicate names?
    # remove DROPped columns before parsing

    for column in df:
        df[column] = df[column].apply(_convert_data_item, args=(null_value,))

    if coldiff > 0:
        for _ in range(coldiff):    # Create empty columns. Pandas does not support df[[None, None]] = [0, 0] or similar hence the loop
            df[None] = float(null_value)
    elif coldiff < 0:
        pass
    
    df.columns = column_names
    return df


def read_raw_dataset(filename_or_io, colnames=[], coltypes=None, ignore_character='@'):
    """ Read in an NM-TRAN dataset with minimal processing. All values will be kept as strings
        by default tries to infer the ColumnTypes from the column names
    """
    df = _read_raw_dataset_from_file(filename_or_io, ignore_character)
    diff_cols = len(df.columns) - len(colnames)
    if diff_cols > 0:
        colnames += [''] * diff_cols
        coltypes += [ColumnType.UNKNOWN] * diff_cols
    elif diff_cols < 0:
        colnames = colnames[0:len(df.columns)]
        coltypes = coltypes[0:len(df.columns)]
    df.columns = colnames
    for label, coltype in zip(df.columns, coltypes):
        df.pharmpy.column_type[label] = coltype 
    return df


def read_dataset(filename_or_io, colnames=[], coltypes=[], drop=[], ignore_character='@', null_value='0'):
    """ A static method to read in an NM-TRAN dataset and return a PharmDataFrame
    """
    df = _read_raw_dataset_from_file(filename_or_io, ignore_character)
    df = _postprocess_data_frame(df, colnames, coltypes, drop, str(null_value))
    return pharmpy.data.PharmDataFrame(df)
