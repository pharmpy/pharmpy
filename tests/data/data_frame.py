import pytest

import pharmpy.data as data


def test_column_type():
    col_type = data.ColumnType.ID
    assert col_type == data.ColumnType.ID
