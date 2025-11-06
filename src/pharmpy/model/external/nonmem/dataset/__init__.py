# Read dataset from file
import warnings
from itertools import chain
from pathlib import Path
from typing import Any, Container, Iterable, Optional, TextIO, cast

from pharmpy import conf
from pharmpy.deps import pandas as pd
from pharmpy.model import DatasetError, DatasetWarning

from .convert import convert, convert_in_place
from .filter import (
    character,
    conjunction,
    filter_schedule,
    mask_in_place,
    negation,
    numeric,
    parse_filter_statements,
)
from .nmtran import SEP, SEP_INPUT, IOFromChunks, NMTRANDataLines, read_NMTRAN_data


def _make_ids_unique(idcol: str, df: pd.DataFrame, columns: Iterable[str | None]):
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


def _idcol(columns: Container[str | None]) -> str | None:
    if 'ID' in columns:
        return 'ID'
    elif 'L1' in columns:
        return 'L1'
    else:
        return None


def read_nonmem_df(
    path_or_io: str | Path | TextIO,
    raw: bool,
    ignore_character: str,
    colnames: Iterable[str],
):
    with NMTRANDataLines(path_or_io, SEP_INPUT, ignore_character) as lines:
        first_line = next(lines)
        first_line_column_count = first_line.count(SEP) + 1
        columns = list(colnames)

        if raw:
            if len(columns) > first_line_column_count:
                columns = columns[:first_line_column_count]
            else:
                columns += [''] * (first_line_column_count - len(columns))

        elif len(columns) > first_line_column_count:
            warnings.warn("There are more columns in $INPUT than in the dataset")

        header_line = SEP.join(columns) + '\n'

        io = IOFromChunks(map(str.encode, chain([header_line, first_line], lines)))
        usecols = list(range(len(columns)))
        df = read_NMTRAN_data(io, header=0, usecols=usecols)

    return df


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

    df = read_nonmem_df(path_or_io, raw, ignore_character, colnames)

    return filter_and_convert_nonmem_dataset_in_place(
        df,
        raw=raw,
        drop=drop,
        null_value=null_value,
        parse_columns=parse_columns,
        ignore=ignore,
        accept=accept,
        dtype=dtype,
        missing_data_token=missing_data_token,
    )


def filter_and_convert_nonmem_dataset_in_place(
    df: pd.DataFrame,
    raw: bool = False,
    drop: Optional[list[bool]] = None,
    null_value: str = '0',
    parse_columns: Optional[Iterable[str]] = None,
    ignore: Optional[list[str]] = None,
    accept: Optional[list[str]] = None,
    dtype: Optional[dict[str, Any]] = None,
    missing_data_token: Optional[str] = None,
):
    columns = df.columns
    idcol = _idcol(columns)

    if drop is None:
        drop = [False] * len(columns)

    non_dropped = [name for name, dropped in zip(columns, drop) if not dropped]
    if len(non_dropped) > len(set(non_dropped)):
        raise KeyError('Column names are not unique')

    missing_data_token = (
        missing_data_token if missing_data_token is not None else conf.missing_data_token
    )

    if ignore and accept:
        raise ValueError("Cannot have both IGNORE and ACCEPT")

    convert_todo = (
        set(parse_columns)
        if parse_columns is not None
        else (set() if raw else set(col for col, dropped in zip(columns, drop) if not dropped))
    )

    if not raw:
        convert_todo.difference_update(("TIME", "DATE", "DAT1", "DAT2", "DAT3"))

    statements = ignore or accept
    if statements is None:
        convert_remaining = list(convert_todo)
    else:
        df, convert_done = _filter_in_place(
            df,
            columns,
            statements,
            convert_todo,
            negation if statements is ignore else lambda x: x,
            conjunction,
            null_value,
            missing_data_token,
        )
        convert_remaining = list(convert_todo.difference(convert_done))

    if convert_remaining:
        convert_in_place(df, convert_remaining, str(null_value), missing_data_token)

    if idcol is not None:
        _make_ids_unique(idcol, df, convert_todo)

        if not raw and all((_ids := df[idcol].astype('int32')) == df[idcol]):
            df[idcol] = _ids

    if not raw:
        # Parse TIME if possible
        if 'TIME' in columns and not any(
            item in columns for item in ['DATE', 'DAT1', 'DAT2', 'DAT3']
        ):
            try:
                convert_in_place(df, ["TIME"], str(null_value), missing_data_token)
            except DatasetError:
                if dtype and 'TIME' in dtype:
                    dtype['TIME'] = 'str'
                pass

    if dtype:
        _columns = set(columns)
        _dtype = {k: v for k, v in dtype.items() if k in _columns}
        if _dtype:
            df = df.astype(_dtype)

    df.reset_index(drop=True, inplace=True)
    return df


def _filter_in_place(
    df, columns, statements, convert_todo, _map, _reduce, null_value, missing_data_token
):
    filters = parse_filter_statements(statements)

    df.columns = list(map(character, columns))
    tmp = df
    blocks = list(filter_schedule(filters))

    for block in blocks:

        if block.convert:
            tmp[list(map(numeric, block.convert))] = convert(
                tmp[list(map(character, block.convert))], str(null_value), missing_data_token
            )

        mask_in_place(tmp, block.filters, _map, _reduce)

    convert_done = set().union(*(block.convert for block in blocks)).intersection(convert_todo)

    convert_init = [
        numeric(column) if column in convert_done else character(column) for column in columns
    ]

    df = cast(pd.DataFrame, tmp[convert_init]).copy(deep=False)
    del tmp
    df.columns = columns

    return df, convert_done
