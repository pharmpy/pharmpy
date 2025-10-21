# Read dataset from file
import re
import warnings
from typing import Iterable, cast

from pharmpy import conf
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model import DatasetError, DatasetWarning

from .nmtran_data import SEP_INPUT, NMTRANDataIO, read_NMTRAN_data
from .nmtran_filter import (
    character,
    conjunction,
    filter_schedule,
    mask_in_place,
    negation,
    numeric,
    parse_filter_statements,
    query,
)


def convert_fortran_number(number_string):
    """This function will try to convert the number_string from the general fortran exponential format
    into an np.float64. It covers "1d1", "1D1", "a+b", "a-b", "+" and "-". All other cases will
    return None to signal that the number_string is not of the special form.

    Move somewhere else. Will be used in output parsing as well
    """
    try:
        y = np.float64(number_string)
        return y
    except (TypeError, ValueError):
        pass

    if number_string == '+' or number_string == '-':
        return float(number_string + "0.0")  # Converts "-" into -0.0

    # Handles formats like "1+1" = 1.0e1
    m = re.match(r'([+\-]?)([^+\-dD]*)([+-])([^+\-dD]*)', number_string)
    if m:
        mantissa_sign = '-' if m.group(1) == '-' else ''
        mantissa = m.group(2)
        exponent_sign = m.group(3)
        exponent = m.group(4)
        return np.float64(mantissa_sign + mantissa + "E" + exponent_sign + exponent)

    # Handles normal cases of using D or d instead of E or e
    if "D" in number_string or "d" in number_string:
        clean_number = number_string.replace("D", "e").replace("d", "e")
        try:
            y = np.float64(clean_number)
            return y
        except (TypeError, ValueError):
            pass

    raise ValueError(f"Could not convert the fortran number {number_string} to float")


def _convert_data_item(x, null_value, missing_data_token):
    if x is None or x == '.' or x == '':
        x = null_value
    if len(x) > 24:
        raise DatasetError("The dataset contains an item that is longer than 24 characters")
    if x == missing_data_token:
        return np.nan
    try:
        converted = convert_fortran_number(x)
    except ValueError as e:
        raise DatasetError(str(e)) from e
    return converted


_convert_data_item_vectorized = np.vectorize(_convert_data_item)


def convert(df, null_value: str, missing_data_token: str):
    return df.apply(_convert_data_item_vectorized, args=(null_value, missing_data_token))


def _make_ids_unique(idcol: str, df: pd.DataFrame, columns: Iterable[str]):
    """Check if id numbers are reused and make renumber. If not simply pass through the dataset."""
    if idcol not in columns:
        return

    id_series = df[idcol]
    id_change = id_series.diff(1) != 0
    if len(id_series[id_change]) != len(id_series.unique()):
        warnings.warn(
            "Dataset contains non-unique id numbers. Renumbering starting from 1",
            DatasetWarning,
        )
        df[idcol] = id_change.cumsum()


def _idcol(df: pd.DataFrame):
    columns = df.columns
    if 'ID' in columns:
        return 'ID'
    elif 'L1' in columns:
        return 'L1'
    else:
        return None


def read_nonmem_dataset(
    path_or_io,
    raw=False,
    ignore_character='#',
    colnames=(),
    drop=None,
    null_value='0',
    parse_columns=None,
    ignore=None,
    accept=None,
    dtype=None,
    missing_data_token=None,
):
    """Read a nonmem dataset from file
     column types will be inferred from the column names

    raw - minimal processing, data will be kept in string format.
    ignore_character
    colnames - List or tuple of names to give each column given in order. Names need to be unique
    drop - A list or tuple of booleans of which columns to drop
    null_value - Value to use for NULL, i.e. empty records or padding
    parse_columns - Only applicable when raw=True. A list of columns to parse.
    ignore/accept - List of ignore/accept expressions

     The following postprocessing operations are done to a non-raw dataset
     1. Convert ordinary floating point numbers to float64
     2. Convert numbers of special fortran format to float64
     3. Convert None, '.', empty string to the NULL value
     4. Convert Inf/NaN properly
     5. Pad with null_token columns if $INPUT has more columns than the dataset
     6. Strip away superfluous columns from the dataset
    """
    if drop is None:
        drop = [False] * len(colnames)

    missing_data_token = (
        missing_data_token if missing_data_token is not None else conf.missing_data_token
    )

    non_dropped = [name for name, dropped in zip(colnames, drop) if not dropped]
    if len(non_dropped) > len(set(non_dropped)):
        raise KeyError('Column names are not unique')

    with NMTRANDataIO(path_or_io, SEP_INPUT, ignore_character) as io:
        df = read_NMTRAN_data(io, header=None)

    assert isinstance(df, pd.DataFrame)

    diff_cols = len(df.columns) - len(colnames)
    if diff_cols > 0:
        df.columns = list(colnames) + [None] * diff_cols
        if not raw:
            # Remove unnamed columns
            df.drop(columns=[None], inplace=True)
    elif diff_cols < 0:
        if raw:
            df.columns = colnames[0 : len(df.columns)]
        else:
            warnings.warn("There are more columns in $INPUT than in the dataset")
            for i in range(abs(diff_cols)):  # Create empty columns.
                df[f'__{i}]'] = str(null_value)  # FIXME assure no name collisions here
            df.columns = colnames
    else:
        df.columns = colnames

    idcol = _idcol(df)

    if ignore and accept:
        raise ValueError("Cannot have both IGNORE and ACCEPT")

    statements = ignore or accept
    if statements is None:
        filters = []
    else:
        filters = list(parse_filter_statements(statements))

    columns = df.columns
    df.columns = list(map(character, columns))
    tmp = df
    blocks = list(filter_schedule(filters))

    for block in blocks:

        if block.convert:
            tmp[list(map(numeric, block.convert))] = convert(
                tmp[list(map(character, block.convert))], str(null_value), missing_data_token
            )

        mask = query(
            tmp, block.filters, negation if statements is ignore else lambda x: x, conjunction
        )
        mask_in_place(tmp, mask)

    convert_todo = (
        set(parse_columns)
        if parse_columns is not None
        else (set() if raw else set(col for col, dropped in zip(columns, drop) if not dropped))
    )

    if not raw:
        convert_todo.difference_update(("TIME", "DATE", "DAT1", "DAT2", "DAT3"))

    convert_done = set().union(*(block.convert for block in blocks)).intersection(convert_todo)
    convert_init = [
        numeric(column) if column in convert_done else character(column) for column in columns
    ]

    df = cast(pd.DataFrame, tmp[convert_init].copy())
    del tmp
    df.columns = columns

    convert_remaining = list(convert_todo.difference(convert_done))
    if convert_remaining:
        df[convert_remaining] = convert(df[convert_remaining], str(null_value), missing_data_token)

    if idcol is not None:
        _make_ids_unique(idcol, df, convert_todo)

        if not raw and all(df[idcol].astype('int32') == df[idcol]):
            df[idcol] = df[idcol].astype('int32')

    if not raw:
        # Parse TIME if possible
        if 'TIME' in df.columns and not any(
            item in df.columns for item in ['DATE', 'DAT1', 'DAT2', 'DAT3']
        ):
            try:
                df[["TIME"]] = convert(df[["TIME"]], str(null_value), missing_data_token)
            except DatasetError:
                if dtype and 'TIME' in dtype:
                    dtype['TIME'] = 'str'
                pass

    if dtype:
        cols = set(df.columns)
        _dtype = {k: v for k, v in dtype.items() if k in cols}
        if _dtype:
            df = df.astype(_dtype)

    return df
