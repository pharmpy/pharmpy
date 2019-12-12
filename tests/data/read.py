import pytest
import pandas as pd
from io import StringIO

import pharmpy.data as data
from pharmpy.data import DatasetError
from pharmpy.data import DatasetWarning
from pharmpy.data import ColumnType


def test_read_nonmem_dataset(testdata):
    path = testdata / 'nonmem' / 'pheno.dta'
    colnames = ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV', 'FA1', 'FA2']
    df = data.read_nonmem_dataset(path, colnames=colnames)
    assert list(df.columns) == colnames
    assert df.pharmpy.column_type['ID'] == data.ColumnType.ID
    assert df['ID'][0] == 1
    assert df['TIME'][2] == 12.5

    raw = data.read_nonmem_dataset(path, colnames=colnames, raw=True)
    assert raw['ID'][0] == '1'
    assert list(df.columns) == colnames
    
    df_drop = data.read_nonmem_dataset(path, colnames=colnames, drop=[False, False, True, False, False, False, True, False])
    assert list(df_drop.columns) == ['ID', 'TIME', 'WGT', 'APGR', 'DV', 'FA2']
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


@pytest.mark.parametrize("s,expected", [
    ( '1.0', 1.0 ),
    ( '+', 0.0 ),
    ( '-', 0.0 ),
    ( '1d1', 10 ),
    ( '1.25D+2', 125 ),
    ( '1+2', 100),
    ( '4-1', 0.4),
    ( '0.25+2', 25),
])
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
