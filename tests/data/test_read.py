from io import StringIO

import pandas as pd
import pytest

import pharmpy.data as data
from pharmpy.data import DatasetError, DatasetWarning


def test_read_nonmem_dataset(testdata):
    path = testdata / 'nonmem' / 'pheno.dta'
    colnames = ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV', 'FA1', 'FA2']
    df = data.read_nonmem_dataset(path, colnames=colnames, ignore_character='@')
    assert list(df.columns) == colnames
    assert df.pharmpy.column_type['ID'] == data.ColumnType.ID
    assert df['ID'][0] == 1
    assert df['TIME'][2] == '12.5'  # FIXME! Should be number

    raw = data.read_nonmem_dataset(path, colnames=colnames, ignore_character='@', raw=True)
    assert raw['ID'][0] == '1'
    assert list(df.columns) == colnames

    raw2 = data.read_nonmem_dataset(
        path, colnames=colnames, ignore_character='@', raw=True, parse_columns=['ID']
    )
    assert raw2['ID'][0] == 1.0

    df_drop = data.read_nonmem_dataset(
        path,
        colnames=colnames,
        ignore_character='@',
        drop=[False, False, True, False, False, False, True, False],
    )
    # FIXME: DROP not decided yet.
    # assert list(df_drop.columns) == ['ID', 'TIME', 'WGT', 'APGR', 'DV', 'FA2']
    pd.testing.assert_series_equal(df_drop['ID'], df['ID'])
    pd.testing.assert_series_equal(df_drop['FA2'], df['FA2'])


def test_data_io(testdata):
    path = testdata / 'nonmem' / 'pheno.dta'
    data_io = data.read.NMTRANDataIO(path, '@')
    assert data_io.read(7) == '      1'
    data_io = data.read.NMTRANDataIO(path, 'I')
    assert data_io.read(13) == '      1    0.'
    data_io = data.read.NMTRANDataIO(path, 'Q')
    assert data_io.read(5) == 'ID TI'


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
    assert data.read.convert_fortran_number(s) == expected


def test_read_small_nonmem_datasets():
    abc = ['A', 'B', 'C']
    df = data.read_nonmem_dataset(StringIO("1,2,3"), colnames=abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    assert list(df.columns) == ['A', 'B', 'C']
    df = data.read_nonmem_dataset(StringIO("1, 2   , 3"), colnames=abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    df = data.read_nonmem_dataset(StringIO("1,,3"), colnames=abc)
    assert list(df.iloc[0]) == [1, 0, 3]
    df = data.read_nonmem_dataset(StringIO("1,,"), colnames=abc)
    assert list(df.iloc[0]) == [1, 0, 0]
    df = data.read_nonmem_dataset(StringIO(",2,4"), colnames=abc)
    assert list(df.iloc[0]) == [0, 2, 4]
    df = data.read_nonmem_dataset(StringIO("1\t2\t3"), colnames=abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    df = data.read_nonmem_dataset(StringIO("1\t2\t"), colnames=abc)
    assert list(df.iloc[0]) == [1, 2, 0]
    df = data.read_nonmem_dataset(StringIO("3 4 6"), colnames=abc)
    assert list(df.iloc[0]) == [3, 4, 6]
    df = data.read_nonmem_dataset(StringIO("3   28   , 341"), colnames=abc)
    assert list(df.iloc[0]) == [3, 28, 341]
    df = data.read_nonmem_dataset(StringIO("  1  2  3  "), colnames=abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    with pytest.raises(DatasetError):
        df = data.read_nonmem_dataset(StringIO("  1  2  3  \n\n"), colnames=abc)
    with pytest.raises(DatasetError):
        df = data.read_nonmem_dataset(StringIO("1\t2 \t3"), colnames=abc)

    # Mismatch length of column_names and data frame
    df = data.read_nonmem_dataset(StringIO("1,2,3"), colnames=abc + ['D'])
    assert list(df.iloc[0]) == [1, 2, 3, 0]
    assert list(df.columns) == ['A', 'B', 'C', 'D']
    df = data.read_nonmem_dataset(StringIO("1,2,3,6"), colnames=abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    assert list(df.columns) == ['A', 'B', 'C']

    # Test null_value
    df = data.read_nonmem_dataset(StringIO("1,2,"), colnames=abc, null_value=9)
    assert list(df.iloc[0]) == [1, 2, 9]


def test_nonmem_dataset_with_nonunique_ids():
    colnames = ['ID', 'DV']
    with pytest.warns(DatasetWarning):
        df = data.read_nonmem_dataset(StringIO("1,2\n2,3\n1,4\n2,5"), colnames=colnames)
    assert list(df[df.pharmpy.id_label]) == [1, 2, 3, 4]
    assert list(df['DV']) == [2, 3, 4, 5]


def test_nonmem_dataset_with_ignore_accept():
    colnames = ['ID', 'DV']
    df = data.read_nonmem_dataset(
        StringIO("1,2\n1,3\n2,4\n2,5"), colnames=colnames, ignore=['DV.EQN.2']
    )
    assert len(df) == 3
    assert list(df.columns) == colnames
    assert list(df.iloc[0]) == [1, 3]
    assert list(df.iloc[1]) == [2, 4]
    assert list(df.iloc[2]) == [2, 5]
    df = data.read_nonmem_dataset(
        StringIO("1,2\n1,3\n2,4\n2,a"), colnames=colnames, ignore=['DV.EQ.a', 'DV.EQN.2']
    )
    assert len(df) == 2
    assert list(df.columns) == colnames
    assert list(df.iloc[0]) == [1, 3]
    assert list(df.iloc[1]) == [2, 4]
    with pytest.raises(DatasetError):
        df = data.read_nonmem_dataset(
            StringIO("1,2\n1,3\n2,4\n2,a"), colnames=colnames, ignore=['DV.EQN.2', 'DV.EQ.a']
        )
    df = data.read_nonmem_dataset(
        StringIO("1,2\n1,3\n2,4\n2,a"), colnames=colnames, ignore=['DV.EQ."a"']
    )
    assert len(df) == 3
    assert list(df.columns) == colnames
    assert list(df.iloc[0]) == [1, 2]
    assert list(df.iloc[1]) == [1, 3]
    assert list(df.iloc[2]) == [2, 4]
    df = data.read_nonmem_dataset(
        StringIO("1,2\n1,3\n2,4\n2,5"), colnames=colnames, accept=['DV.EQN.2']
    )
    assert len(df) == 1
    assert list(df.columns) == colnames
    assert list(df.iloc[0]) == [1, 2]
    df = data.read_nonmem_dataset(
        StringIO("1,2\n1,3\n2,4\n2,5"), colnames=colnames, ignore=['ID 2']
    )
    assert len(df) == 2
    assert list(df.iloc[0]) == [1, 2]
    assert list(df.iloc[1]) == [1, 3]
