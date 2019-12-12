# Read dataset from file
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

        if re.search(r'^[ \t]*\n$', contents, re.MULTILINE):       # Blank lines
            raise DatasetError("The dataset contains one or more blank lines. This is not allowed by NM-TRAN without the BLANKOK option")

        super().__init__(contents)


def convert_fortran_number(number_string):
    """This function will try to convert the number_string from the general fortran exponential format
       into an np.float64. It covers "1d1", "1D1", "a+b", "a-b", "+" and "-". All other cases will return None to
       signal that the number_string is not of the special form.

       Move somewhere else. Will be used in output parsing as well
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
    elif colname == 'DV':
        return ColumnType.DV
    else:
        return ColumnType.UNKNOWN


def read_nonmem_dataset(path_or_io, raw=False, ignore_character='@', colnames=tuple(), coltypes=None, drop=None, null_value='0', parse_columns=tuple()):
    """Read a nonmem dataset from file
        column types will be inferred from the column names

       raw - minimal processing, data will be kept in string format.
       ignore_character
       colnames - List or tuple of names to give each column given in order. Names need to be unique
       drop - A list or tuple of booleans of which columns to drop
       null_value - Value to use for NULL, i.e. empty records or padding
       parse_columns - Only applicable when raw=True. A list of columns to parse.
      
        The following postprocessing operations are done to a non-raw dataset
        1. Convert ordinary floating point numbers to float64
        2. Convert numbers of special fortran format to float64
        3. Convert None, '.', empty string to the NULL value
        4. Convert Inf/NaN properly
        5. Pad with null_token columns if $INPUT has more columns than the dataset
        6. Strip away superfluous columns from the dataset
    """
    if len(colnames) > len(set(colnames)):
        raise KeyError('Column names are not unique')

    file_io = NMTRANDataIO(path_or_io, ignore_character)
    df = pd.read_table(file_io, sep=r' *, *| *[\t] *| +', na_filter=False, header=None, engine='python', quoting=3, dtype=np.object)
    df = pharmpy.data.PharmDataFrame(df)

    diff_cols = len(df.columns) - len(colnames)
    if diff_cols > 0:
        df.columns = list(colnames) + [None] * diff_cols
        if not raw:
            # Remove unnamed columns
            df.drop(columns=[None], inplace=True)
    elif diff_cols < 0:
        if raw:
            df.columns = colnames[0:len(df.columns)]
        else:
            for i in range(abs(diff_cols)):    # Create empty columns.
                df[f'__{i}]'] = null_value       # FIXME assure no name collisions here
            df.columns = colnames
    else:
        df.columns = colnames

    if coltypes is None:
        for label in df.columns:
            df.pharmpy.column_type[label] = infer_column_type(label)
    else:
        if len(coltypes) < len(df.columns):
            coltypes += [data.ColumnType.UNKNOWN] * (len(df.columns) - len(coltypes))
        df.pharmpy.column_type[list(df.columns)] = coltypes

    if drop:
        indices_to_drop = [i for i, x in enumerate(drop) if not x]
        df = df.iloc[:, indices_to_drop].copy()

    if not raw:
        parse_columns = list(df.columns)
    for column in parse_columns:
        df[column] = df[column].apply(_convert_data_item, args=(str(null_value),))

    return df
