from io import StringIO
import pytest

import pharmpy.plugins.nonmem.input as inp
from pharmpy.data import DatasetError
from pharmpy.data import DatasetWarning


def test_data_io(pheno_data):
    data_io = inp.NMTRANDataIO(pheno_data, '@')
    assert data_io.read(7) == '      1'
    data_io = inp.NMTRANDataIO(pheno_data, 'I')
    assert data_io.read(13) == '      1    0.'
    data_io = inp.NMTRANDataIO(pheno_data, 'Q')
    assert data_io.read(5) == 'ID TI'


def test_data_read_data_frame():
    abc = ['A', 'B', 'C']
    mi = inp.ModelInput
    df = mi.read_dataset(StringIO("1,2,3"), abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    assert list(df.columns) == ['A', 'B', 'C']
    df = mi.read_dataset(StringIO("1, 2   , 3"), abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    df = mi.read_dataset(StringIO("1,,3"), abc)
    assert list(df.iloc[0]) == [1, 0, 3]
    df = mi.read_dataset(StringIO("1,,"), abc)
    assert list(df.iloc[0]) == [1, 0, 0]
    df = mi.read_dataset(StringIO(",2,4"), abc)
    assert list(df.iloc[0]) == [0, 2, 4]
    df = mi.read_dataset(StringIO("1\t2\t3"), abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    df = mi.read_dataset(StringIO("1\t2\t"), abc)
    assert list(df.iloc[0]) == [1, 2, 0]
    df = mi.read_dataset(StringIO("3 4 6"), abc)
    assert list(df.iloc[0]) == [3, 4, 6]
    df = mi.read_dataset(StringIO("3   28   , 341"), abc)
    assert list(df.iloc[0]) == [3, 28, 341]
    df = mi.read_dataset(StringIO("  1  2  3  "), abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    df = mi.read_dataset(StringIO("  1  2  3  \n\n"), abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    with pytest.raises(DatasetError):
        df = mi.read_dataset(StringIO("1\t2 \t3"), abc)

    # Mismatch length of column_names and data frame
    df = mi.read_dataset(StringIO("1,2,3"), abc + ['D'])
    assert list(df.iloc[0]) == [1, 2, 3, 0]
    assert list(df.columns) == ['A', 'B', 'C', 'D']
    with pytest.warns(DatasetWarning):
        df = mi.read_dataset(StringIO("1,2,3,6"), abc)
    assert list(df.iloc[0]) == [1, 2, 3]
    assert list(df.columns) == ['A', 'B', 'C']

    # Test null_value
    df = mi.read_dataset(StringIO("1,2,"), abc, null_value=9)
    assert list(df.iloc[0]) == [1, 2, 9]


def test_data_read(pheno):
    df = pheno.input.dataset
    assert list(df.iloc[1]) == [1.0, 2.0, 0.0, 1.4, 7.0, 17.3, 0.0, 0.0]
    assert list(df.columns) == ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV', 'FA1', 'FA2']
