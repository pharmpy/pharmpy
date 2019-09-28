import pytest

import pharmpy.data as data


def test_column_type():
    col_type = data.ColumnType.ID
    assert col_type == data.ColumnType.ID


def test_data_frame():
    df = data.PharmDataFrame({'ID': [1, 1, 2, 2], 'DV': [0.1, 0.2, 0.5, 0.6]})
    assert list(df.columns) == ['ID', 'DV']


def test_accessor_get_set_column_type():
    df = data.PharmDataFrame({'ID': [1, 1, 2, 2], 'DV': [0.1, 0.2, 0.5, 0.6]})
    assert df.pharmpy.get_column_type('ID') == data.ColumnType.UNKNOWN
    df.pharmpy.set_column_type('ID', data.ColumnType.ID)
    df.pharmpy.set_column_type('DV', data.ColumnType.DV)
    assert df.pharmpy.get_column_type('ID') == data.ColumnType.ID
    assert df.pharmpy.get_column_type('DV') == data.ColumnType.DV
    df2 = df.copy()
    assert df2.pharmpy.get_column_type('ID') == data.ColumnType.ID
    assert df2.pharmpy.get_column_type('DV') == data.ColumnType.DV
