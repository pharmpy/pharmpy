from io import StringIO
import pytest

from pharmpy.data import ColumnType


def test_data_read(pheno):
    df = pheno.input.dataset
    assert list(df.iloc[1]) == [1.0, 2.0, 0.0, 1.4, 7.0, 17.3, 0.0, 0.0]
    assert list(df.columns) == ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV', 'FA1', 'FA2']


def test_read_raw_dataset(pheno):
    df = pheno.input.raw_dataset
    assert list(df.iloc[0]) == ['1', '0.', '25.0', '1.4', '7', '0', '1', '1']
    assert list(df.columns) == ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV', 'FA1', 'FA2']
    assert df.pharmpy.column_type['ID'] == ColumnType.ID
