import pytest
import pandas as pd

import pharmpy.data as data


@pytest.fixture
def df():
    df = data.PharmDataFrame({'ID': [1, 1, 2, 2], 'DV': [0.1, 0.2, 0.5, 0.6], 'WGT': [70, 72, 75, 75], 'HGT': [185, 185, 160, 160]})
    df.pharmpy.column_type[('ID', 'DV', 'WGT', 'HGT')] = [data.ColumnType.ID, data.ColumnType.DV, data.ColumnType.COVARIATE, data.ColumnType.COVARIATE]
    return df


def test_column_type():
    col_type = data.ColumnType.ID
    assert col_type == data.ColumnType.ID
    assert 'ID' in repr(col_type)
    assert col_type.max_one


def test_data_frame():
    df = data.PharmDataFrame({'ID': [1, 1, 2, 2], 'DV': [0.1, 0.2, 0.5, 0.6]})
    assert list(df.columns) == ['ID', 'DV']


def test_accessor_get_set_column_type():
    df = data.PharmDataFrame({'ID': [1, 1, 2, 2], 'DV': [0.1, 0.2, 0.5, 0.6], 'WGT': [70, 70, 75, 75], 'HGT': [185, 185, 160, 160]})
    assert df.pharmpy.column_type['ID'] == data.ColumnType.UNKNOWN
    assert df.pharmpy.column_type[('ID', 'DV')] == [data.ColumnType.UNKNOWN, data.ColumnType.UNKNOWN]
    df.pharmpy.column_type['ID'] = data.ColumnType.ID
    df.pharmpy.column_type['DV'] = data.ColumnType.DV
    assert df.pharmpy.column_type['ID'] == data.ColumnType.ID
    assert df.pharmpy.column_type['DV'] == data.ColumnType.DV
    assert df.pharmpy.column_type[['ID', 'DV']] == [data.ColumnType.ID, data.ColumnType.DV]
    df.pharmpy.column_type[['HGT', 'WGT']] = data.ColumnType.COVARIATE
    assert df.pharmpy.labels_by_type[data.ColumnType.COVARIATE] == ['WGT', 'HGT']
    df2 = df.copy()
    assert df2.pharmpy.column_type['ID'] == data.ColumnType.ID
    assert df2.pharmpy.column_type['DV'] == data.ColumnType.DV


    assert df2.pharmpy.labels_by_type[data.ColumnType.ID] == ['ID']
    assert df2.pharmpy.labels_by_type[data.ColumnType.DV] == ['DV']

    assert df2.pharmpy.id_label == 'ID'

    df2.pharmpy.column_type[['ID', 'DV']] = (data.ColumnType.COVARIATE, data.ColumnType.IDV)
    assert df2.pharmpy.column_type[['ID', 'DV']] == [data.ColumnType.COVARIATE, data.ColumnType.IDV]


def test_time_varying_covariates(df):
    assert df.pharmpy.time_varying_covariates == ['WGT']
    pd.testing.assert_frame_equal(df.pharmpy.covariate_baselines, pd.DataFrame({'WGT': [70, 75], 'HGT': [185, 160]}, index=pd.Int64Index([1, 2], name='ID')))
