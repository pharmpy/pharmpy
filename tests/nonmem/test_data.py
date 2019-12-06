from io import StringIO
import pytest

import pharmpy.plugins.nonmem.data as data
from pharmpy.data import DatasetError
from pharmpy.data import DatasetWarning
from pharmpy.data import ColumnType


def test_data_io(pheno_data):
    data_io = data.NMTRANDataIO(pheno_data, '@')
    assert data_io.read(7) == '      1'
    data_io = data.NMTRANDataIO(pheno_data, 'I')
    assert data_io.read(13) == '      1    0.'
    data_io = data.NMTRANDataIO(pheno_data, 'Q')
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
    assert data.convert_fortran_number(s) == expected

def test_read_dataset():
    abc = ['A', 'B', 'C']
    df = data.read_dataset(StringIO("1,2,3"), abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    assert list(df.columns) == ['A', 'B', 'C']
    df = data.read_dataset(StringIO("1, 2   , 3"), abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    df = data.read_dataset(StringIO("1,,3"), abc)
    assert list(df.iloc[0]) == [1, 0, 3]
    df = data.read_dataset(StringIO("1,,"), abc)
    assert list(df.iloc[0]) == [1, 0, 0]
    df = data.read_dataset(StringIO(",2,4"), abc)
    assert list(df.iloc[0]) == [0, 2, 4]
    df = data.read_dataset(StringIO("1\t2\t3"), abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    df = data.read_dataset(StringIO("1\t2\t"), abc)
    assert list(df.iloc[0]) == [1, 2, 0]
    df = data.read_dataset(StringIO("3 4 6"), abc)
    assert list(df.iloc[0]) == [3, 4, 6]
    df = data.read_dataset(StringIO("3   28   , 341"), abc)
    assert list(df.iloc[0]) == [3, 28, 341]
    df = data.read_dataset(StringIO("  1  2  3  "), abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    df = data.read_dataset(StringIO("  1  2  3  \n\n"), abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    with pytest.raises(DatasetError):
        df = data.read_dataset(StringIO("1\t2 \t3"), abc)

    # Mismatch length of column_names and data frame
    df = data.read_dataset(StringIO("1,2,3"), abc + ['D'])
    assert list(df.iloc[0]) == [1, 2, 3, 0]
    assert list(df.columns) == ['A', 'B', 'C', 'D']
    with pytest.warns(DatasetWarning):
        df = data.read_dataset(StringIO("1,2,3,6"), abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    assert list(df.columns) == ['A', 'B', 'C']

    # Test null_value
    df = data.read_dataset(StringIO("1,2,"), abc, null_value=9)
    assert list(df.iloc[0]) == [1, 2, 9]
