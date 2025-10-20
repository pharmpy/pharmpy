from io import StringIO

import pytest

from pharmpy.deps import pandas as pd
from pharmpy.model import DatasetError, DatasetWarning
from pharmpy.model.external.nonmem.dataset import (
    convert_fortran_number,
    read_nonmem_dataset,
)
from pharmpy.model.external.nonmem.nmtran_data import SEP_INPUT, NMTRANDataIO


def test_read_nonmem_dataset(testdata):
    path = testdata / 'nonmem' / 'pheno.dta'
    colnames = ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV', 'FA1', 'FA2']
    df = read_nonmem_dataset(path, colnames=colnames, ignore_character='@')
    assert list(df.columns) == colnames
    assert df['ID'][0] == 1
    assert df['TIME'][2] == 12.5

    raw = read_nonmem_dataset(path, colnames=colnames, ignore_character='@', raw=True)
    assert raw['ID'][0] == '1'
    assert list(df.columns) == colnames

    raw2 = read_nonmem_dataset(
        path, colnames=colnames, ignore_character='@', raw=True, parse_columns=['ID']
    )
    assert raw2['ID'][0] == 1.0

    df_drop = read_nonmem_dataset(
        path,
        colnames=colnames,
        ignore_character='@',
        drop=[False, False, True, False, False, False, True, False],
    )
    pd.testing.assert_series_equal(df_drop['ID'], df['ID'])
    pd.testing.assert_series_equal(df_drop['FA2'], df['FA2'])


def test_data_io_alpha(testdata):
    path = testdata / 'nonmem' / 'pheno.dta'
    with NMTRANDataIO(path, SEP_INPUT, '@') as data_io:
        assert data_io.read(7) == b'1\t0.\t25'


def test_data_io_i(testdata):
    path = testdata / 'nonmem' / 'pheno.dta'
    with NMTRANDataIO(path, SEP_INPUT, 'I') as data_io:
        assert data_io.read(13) == b'1\t0.\t25.0\t1.4'


def test_data_io_q(testdata):
    path = testdata / 'nonmem' / 'pheno.dta'
    with NMTRANDataIO(path, SEP_INPUT, 'Q') as data_io:
        assert data_io.read(5) == b'ID\tTI'


@pytest.mark.parametrize(
    "s,expected",
    [
        ('1.0', 1.0),
        ('+', 0.0),
        ('-', 0.0),
        ('1d1', 10),
        ('1.25D+2', 125),
        ('1+2', 100),
        ('4-1', 0.4),
        ('0.25+2', 25),
    ],
)
def test_convert_fortran_number(s, expected):
    assert convert_fortran_number(s) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    (
        ("1,2,3", [1, 2, 3]),
        ("1, 2   , 3", [1, 2, 3]),
        ("1,,3", [1, 0, 3]),
        ("1,,", [1, 0, 0]),
        (",2,4", [0, 2, 4]),
        ("1\t2\t3", [1, 2, 3]),
        ("3 4 6", [3, 4, 6]),
        ("3   28   , 341", [3, 28, 341]),
        ("  1  2  3  ", [1, 2, 3]),
        ("1,2,3,6", [1, 2, 3]),
    ),
)
def test_read_nonmem_dataset_line(line, expected):
    abc = ['A', 'B', 'C']
    df = read_nonmem_dataset(StringIO(line), colnames=abc)
    assert list(df.columns) == ['A', 'B', 'C']
    assert list(df.iloc[0]) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    (
        ("1\t2\t", [1, 2, 0]),
        ("1\t2\n", [1, 2, 0]),
        ("1\t2", [1, 2, 0]),
    ),
)
def test_read_nonmem_dataset_too_many_columns_in_input(line: str, expected: list[int]):
    abc = ['A', 'B', 'C']
    with pytest.warns(UserWarning):
        df = read_nonmem_dataset(StringIO(line), colnames=abc)
    assert list(df.iloc[0]) == expected


def test_read_nonmem_dataset_empty_line():
    abc = ['A', 'B', 'C']
    with pytest.raises(DatasetError):
        read_nonmem_dataset(StringIO("  1  2  3  \n\n"), colnames=abc)


def test_read_nonmem_dataset_tab_preceded_by_space():
    abc = ['A', 'B', 'C']
    with pytest.raises(DatasetError):
        read_nonmem_dataset(StringIO("1\t2 \t3"), colnames=abc)


def test_read_nonmem_dataset_header_too_long():
    abc = ['A', 'B', 'C']
    # Mismatch length of column_names and data frame
    with pytest.warns(UserWarning):
        df = read_nonmem_dataset(StringIO("1,2,3"), colnames=abc + ['D'])
    assert list(df.iloc[0]) == [1, 2, 3, 0]
    assert list(df.columns) == ['A', 'B', 'C', 'D']


def test_read_nonmem_dataset_null_value():
    abc = ['A', 'B', 'C']
    # Test null_value
    df = read_nonmem_dataset(StringIO("1,2,"), colnames=abc, null_value=9)
    assert list(df.iloc[0]) == [1, 2, 9]


def test_nonmem_dataset_with_nonunique_ids():
    colnames = ['ID', 'DV']
    with pytest.warns(DatasetWarning):
        df = read_nonmem_dataset(StringIO("1,2\n2,3\n1,4\n2,5"), colnames=colnames)
    assert list(df['ID']) == [1, 2, 3, 4]
    assert list(df['DV']) == [2, 3, 4, 5]


def test_nonmem_dataset_with_ignore_accept_case_1():
    colnames = ['ID', 'DV']
    df = read_nonmem_dataset(StringIO("1,2\n1,3\n2,4\n2,5"), colnames=colnames, ignore=['DV.EQN.2'])
    assert len(df) == 3
    assert list(df.columns) == colnames
    assert list(df.iloc[0]) == [1, 3]
    assert list(df.iloc[1]) == [2, 4]
    assert list(df.iloc[2]) == [2, 5]


def test_nonmem_dataset_with_ignore_accept_case_2():
    colnames = ['ID', 'DV']
    df = read_nonmem_dataset(
        StringIO("1,2\n1,3\n2,4\n2,a"), colnames=colnames, ignore=['DV.EQ.a', 'DV.EQN.2']
    )
    assert len(df) == 2
    assert list(df.columns) == colnames
    assert list(df.iloc[0]) == [1, 3]
    assert list(df.iloc[1]) == [2, 4]


def test_nonmem_dataset_with_ignore_accept_case_3():
    colnames = ['ID', 'DV']
    with pytest.raises(DatasetError):
        read_nonmem_dataset(
            StringIO("1,2\n1,3\n2,4\n2,a"), colnames=colnames, ignore=['DV.EQN.2', 'DV.EQ.a']
        )


def test_nonmem_dataset_with_ignore_accept_case_4():
    colnames = ['ID', 'DV']
    df = read_nonmem_dataset(
        StringIO("1,2\n1,3\n2,4\n2,a"), colnames=colnames, ignore=['DV.EQ."a"']
    )
    assert len(df) == 3
    assert list(df.columns) == colnames
    assert list(df.iloc[0]) == [1, 2]
    assert list(df.iloc[1]) == [1, 3]
    assert list(df.iloc[2]) == [2, 4]


def test_nonmem_dataset_with_ignore_accept_case_5():
    colnames = ['ID', 'DV']
    df = read_nonmem_dataset(StringIO("1,2\n1,3\n2,4\n2,5"), colnames=colnames, accept=['DV.EQN.2'])
    assert len(df) == 1
    assert list(df.columns) == colnames
    assert list(df.iloc[0]) == [1, 2]


def test_nonmem_dataset_with_ignore_accept_case_6():
    colnames = ['ID', 'DV']
    df = read_nonmem_dataset(StringIO("1,2\n1,3\n2,4\n2,5"), colnames=colnames, ignore=['ID 2'])
    assert len(df) == 2
    assert list(df.iloc[0]) == [1, 2]
    assert list(df.iloc[1]) == [1, 3]


def test_nonmem_dataset_with_numeric_ignore_filter_excluding_character_data():
    colnames = ['ID', 'DV']
    df = read_nonmem_dataset(
        StringIO("1,2\n1,3\n2,4\n2,a\n7,9"), colnames=colnames, ignore=['ID.EQN.2', 'DV.EQN.2']
    )
    # NOTE: This works because ID.EQN.2 excludes the row that contains character data in DV.
    assert len(df) == 2
    assert list(df.columns) == colnames
    assert list(df.iloc[0]) == [1, 3]
    assert list(df.iloc[1]) == [7, 9]


def test_nonmem_dataset_with_numeric_accept_filter_excluding_character_data():
    colnames = ['ID', 'DV']
    df = read_nonmem_dataset(
        StringIO("1,2\n1,3\n2,4\n2,a\n7,9"), colnames=colnames, accept=['ID.NEN.2', 'DV.NEN.2']
    )
    # NOTE: This works because ID.NEN.2 excludes the row that contains character data in DV.
    assert len(df) == 2
    assert list(df.columns) == colnames
    assert list(df.iloc[0]) == [1, 3]
    assert list(df.iloc[1]) == [7, 9]


def test_nonmem_dataset_with_interleaved_string_ignore_filter():
    colnames = ['ID', 'DV']
    df = read_nonmem_dataset(
        StringIO("1,2\n1,3\n2,4\n2,a\n7,9"),
        colnames=colnames,
        ignore=['ID.LE.1', 'DV.EQ.9', 'ID.LE.2'],
    )
    # NOTE: This works because DV.EQ.9 does not require conversion of the DV column.
    assert len(df) == 0
    assert list(df.columns) == colnames


def test_nonmem_dataset_with_interleaved_string_accept_filter():
    colnames = ['ID', 'DV']
    df = read_nonmem_dataset(
        StringIO("1,2\n1,3\n2,4\n2,a\n7,9"),
        colnames=colnames,
        accept=['ID.GT.1', 'DV.NE.9', 'ID.GT.2'],
    )
    # NOTE: This works because DV.NE.9 does not require conversion of the DV column.
    assert len(df) == 0
    assert list(df.columns) == colnames


def test_nonmem_dataset_with_interleaved_numeric_ignore_filter():
    colnames = ['ID', 'DV']

    with pytest.raises(DatasetError):
        # NOTE: This does not work because DV.EQN.9 is before ID.LE.2 which
        # would exclude the character data in DV.
        read_nonmem_dataset(
            StringIO("1,2\n1,3\n2,4\n2,a\n7,9"),
            colnames=colnames,
            ignore=['ID.LE.1', 'DV.EQN.9', 'ID.LE.2'],
        )


def test_nonmem_dataset_with_interleaved_numeric_accept_filter():
    colnames = ['ID', 'DV']

    with pytest.raises(DatasetError):
        # NOTE: This does not work because DV.NEN.9 is before ID.GT.2 which
        # would exclude the character data in DV.
        read_nonmem_dataset(
            StringIO("1,2\n1,3\n2,4\n2,a\n7,9"),
            colnames=colnames,
            accept=['ID.GT.1', 'DV.NEN.9', 'ID.GT.2'],
        )


def test_nonmem_dataset_with_interleaved_numeric_ignore_filter_and_final_string_filter():
    colnames = ['ID', 'DV']

    with pytest.raises(DatasetError):
        # NOTE: This does not work because DV.GE.9 is before ID.EQ.2 which
        # would exclude the character data in DV.
        read_nonmem_dataset(
            StringIO("1,2\n1,3\n2,4\n2,a\n7,9"),
            colnames=colnames,
            ignore=['ID.LE.1', 'DV.GE.9', 'ID.EQ.2'],
        )


def test_nonmem_dataset_with_interleaved_numeric_accept_filter_and_final_string_filter():
    colnames = ['ID', 'DV']

    with pytest.raises(DatasetError):
        # NOTE: This does not work because DV.LT.9 is before ID.NE.2 which
        # would exclude the character data in DV.
        read_nonmem_dataset(
            StringIO("1,2\n1,3\n2,4\n2,a\n7,9"),
            colnames=colnames,
            accept=['ID.GT.1', 'DV.LT.9', 'ID.NE.2'],
        )


def test_nonmem_dataset_with_blocks():
    colnames = ['ID', 'DV']

    df = read_nonmem_dataset(
        StringIO(
            """0 a
               1 b
               2 c
               3 d
               4 e
               5 987
               6 -123
               7 h
               8 i
               9 j
              10 k
              11 l"""
        ),
        colnames=colnames,
        accept=['DV.NE."e"', 'DV.NE.h', 'ID.GE.4', 'ID.LE.7', 'DV.LT.0'],
    )

    assert len(df) == 1
    assert list(df.columns) == colnames
    assert list(df.iloc[0]) == [6, -123]
