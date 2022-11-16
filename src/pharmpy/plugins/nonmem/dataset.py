# Read dataset from file
import re
import warnings
from io import StringIO

from lark import Lark

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model import DatasetError, DatasetWarning, data


class NMTRANDataIO(StringIO):
    """An IO class that is a prefilter for pandas.read_table.
    Things that must be done before using pandas will be done here.
    Currently it takes care of filtering out ignored rows and handles special delimiter cases
    """

    def __init__(self, filename_or_io, ignore_character='#'):
        """filename_or_io is a string with a path, a path object or any IO object, i.e. StringIO"""
        if not ignore_character:
            ignore_character = '#'
        if hasattr(filename_or_io, 'read'):
            contents = filename_or_io.read()
        else:
            with open(str(filename_or_io), 'r', encoding='latin-1') as datafile:
                contents = datafile.read()  # All variations of newlines are converted into \n

        if ignore_character == '@':
            # FIXME: Does this really handle the final line with no new line?
            comment_regexp = re.compile(r'^[ \t]*[A-Za-z#@].*\n', re.MULTILINE)
        else:
            comment_regexp = re.compile('^[' + ignore_character + '].*\n', re.MULTILINE)
        contents = re.sub(comment_regexp, '', contents)

        if re.search(r' \t', contents):  # Space before TAB not allowed (see documentation)
            raise DatasetError(
                "The dataset contains a TAB preceeded by a space, "
                "which is not allowed by NM-TRAN"
            )

        if re.search(r'^[ \t]*\n$', contents, re.MULTILINE):  # Blank lines
            raise DatasetError(
                "The dataset contains one or more blank lines. This is not "
                "allowed by NM-TRAN without the BLANKOK option"
            )

        super().__init__(contents)


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
    try:
        converted = convert_fortran_number(x)
    except ValueError as e:
        raise DatasetError(str(e)) from e
    if converted in data.conf.na_values:
        return np.nan
    return converted


def _make_ids_unique(df, columns):
    """Check if id numbers are reused and make renumber. If not simply pass through the dataset."""
    if 'ID' in df.columns:
        id_label = 'ID'
    elif 'L1' in df.columns:
        id_label = 'L1'
    else:
        return df
    if id_label in columns:
        id_series = df[id_label]
        id_change = id_series.diff(1) != 0
        if len(id_series[id_change]) != len(id_series.unique()):
            warnings.warn(
                "Dataset contains non-unique id numbers. Renumbering starting from 1",
                DatasetWarning,
            )
            df[id_label] = id_change.cumsum()
    return df


def _filter_ignore_accept(df, ignore, accept, null_value):
    if ignore and accept:
        raise ValueError("Cannot have both IGNORE and ACCEPT")

    if not ignore and not accept:
        return df

    statements = ignore if ignore else accept

    grammar = r'''
        start: column skip1? (operator skip2?)? expr
        column: COLNAME
        COLNAME: /\w+/
        skip1: WS
        skip2: WS
        WS: /\s+/
        operator: OP_EQ | OP_STR_EQ | OP_NE | OP_STR_NE | OP_LT | OP_GT | OP_LT_EQ | OP_GT_EQ
        OP_EQ    : ".EQN."
        OP_STR_EQ: ".EQ." | "==" | "="
        OP_NE    : ".NEN."
        OP_STR_NE: ".NE." | "/="
        OP_LT    : ".LT." | "<"
        OP_GT    : ".GT." | ">"
        OP_LT_EQ : ".LE." | "<="
        OP_GT_EQ : ".GE." | ">="
        expr: EXPR | QEXPR
        EXPR  : /[^"',;()=<>\/.\s][^"',;()=\s]*/
        QEXPR : /"[^"]*"/
              | /'[^']*'/
    '''
    parser = Lark(
        grammar,
        start='start',
        parser='lalr',
        lexer='contextual',
        propagate_positions=False,
        maybe_placeholders=False,
        debug=False,
        cache=True,
    )
    for s in statements:
        tree = parser.parse(s)
        column = ''
        expr = ''
        operator = '=='
        operator_type = str
        for st in tree.iter_subtrees():
            if st.data == 'column':
                column = str(st.children[0])
            elif st.data == 'expr':
                expr = str(st.children[0])
            elif st.data == 'operator':
                operator_token = st.children[0]
                tp = operator_token.type  # pyright: ignore [reportGeneralTypeIssues]
                if tp == 'OP_EQ':
                    operator = '=='
                    operator_type = float
                elif tp == 'OP_NE':
                    operator = '!='
                    operator_type = float
                elif tp == 'OP_LT':
                    operator = '<'
                    operator_type = float
                elif tp == 'OP_GT':
                    operator = '>'
                    operator_type = float
                elif tp == 'OP_LT_EQ':
                    operator = '<='
                    operator_type = float
                elif tp == 'OP_GT_EQ':
                    operator = '>='
                    operator_type = float
                elif tp == 'OP_STR_EQ':
                    operator = '=='
                    operator_type = str
                elif tp == 'OP_STR_NE':
                    operator = '!='
                    operator_type = str
        if len(expr) >= 3 and (
            (expr.startswith("'") and expr.endswith("'"))
            or (expr.startswith('"') and expr.endswith('"'))
        ):
            expr = expr[1:-1]

        if operator_type == str:
            expression = f'{column} {operator} "{expr}"'
            if ignore:
                expression = 'not(' + expression + ')'
            df.query(expression, inplace=True)
        else:
            # Need to temporary convert column. Refer to NONMEM fileformat documentation
            # for further information.
            # Using a name with spaces since this cannot collide with other NONMEM names
            magic_colname = 'a a'
            df[magic_colname] = df[column].apply(_convert_data_item, args=(str(null_value),))
            expression = f'`{magic_colname}` {operator} {expr}'
            if ignore:
                expression = 'not(' + expression + ')'
            df.query(expression, inplace=True)
            df.drop(labels=magic_colname, axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def read_nonmem_dataset(
    path_or_io,
    raw=False,
    ignore_character='#',
    colnames=(),
    drop=None,
    null_value='0',
    parse_columns=(),
    ignore=None,
    accept=None,
    dtype=None,
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

    non_dropped = [name for name, dropped in zip(colnames, drop) if not dropped]
    if len(non_dropped) > len(set(non_dropped)):
        raise KeyError('Column names are not unique')

    file_io = NMTRANDataIO(path_or_io, ignore_character)
    df = pd.read_table(
        file_io,
        sep=r' *, *| *[\t] *| +',
        na_filter=False,
        header=None,
        engine='python',
        quoting=3,
        dtype=object,
        index_col=False,
    )

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

    df = _filter_ignore_accept(df, ignore, accept, null_value)

    if not raw:
        parse_columns = [col for col, dropped in zip(df.columns, drop) if not dropped]
        parse_columns = [
            x for x in parse_columns if x not in ['TIME', 'DATE', 'DAT1', 'DAT2', 'DAT3']
        ]
    for column in parse_columns:
        df[column] = df[column].apply(_convert_data_item, args=(str(null_value),))
    df = _make_ids_unique(df, parse_columns)

    if not raw:
        # Make ID int if possible
        if 'ID' in df.columns:
            idcol = 'ID'
        elif 'L1' in df.columns:
            idcol = 'L1'
        else:
            idcol = None
        if idcol:
            if all(df[idcol].astype('int32') == df[idcol]):
                df[idcol] = df[idcol].astype('int32')

        # Parse TIME if possible
        if 'TIME' in df.columns and not any(
            item in df.columns for item in ['DATE', 'DAT1', 'DAT2', 'DAT3']
        ):
            try:
                df['TIME'] = df['TIME'].apply(_convert_data_item, args=(str(null_value),))
            except DatasetError:
                pass

    if dtype:
        for column in df.columns:
            if column in dtype:
                df[column] = df[column].astype(dtype[column])

    return df
