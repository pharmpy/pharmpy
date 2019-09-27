import pytest

import pharmpy.data as data


def test_column_type():
    col_type = data.ColumnType.ID
    assert col_type == data.ColumnType.ID


def test_data_frame():
    df = data.PharmDataFrame({'ID': [1, 1, 2, 2], 'DV': [0.1, 0.2, 0.5, 0.6]})
    assert list(df.columns) == ['ID', 'DV']
