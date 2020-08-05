import copy

import pandas as pd
import pytest

import pharmpy.data as data


@pytest.fixture
def df():
    df = data.PharmDataFrame({'ID': [1, 1, 2, 2], 'DV': [0.1, 0.2, 0.5, 0.6],
                              'WGT': [70, 72, 75, 75], 'HGT': [185, 185, 160, 160]})
    df.pharmpy.column_type[('ID', 'DV', 'WGT', 'HGT')] = [
        data.ColumnType.ID, data.ColumnType.DV, data.ColumnType.COVARIATE,
        data.ColumnType.COVARIATE]
    return df


def test_column_type():
    col_type = data.ColumnType.ID
    assert col_type == data.ColumnType.ID
    assert 'ID' in repr(col_type)
    assert col_type.max_one
    ct2 = data.ColumnType.UNKNOWN
    assert ct2 == data.ColumnType.UNKNOWN
    assert 'UNKNOWN' in repr(ct2)
    assert not ct2.max_one


def test_data_frame():
    df = data.PharmDataFrame({'ID': [1, 1, 2, 2], 'DV': [0.1, 0.2, 0.5, 0.6]})
    assert list(df.columns) == ['ID', 'DV']


def test_accessor_get_set_column_type():
    df = data.PharmDataFrame({'ID': [1, 1, 2, 2], 'DV': [0.1, 0.2, 0.5, 0.6],
                              'WGT': [70, 70, 75, 75], 'HGT': [185, 185, 160, 160]})
    assert df.pharmpy.column_type['ID'] == data.ColumnType.UNKNOWN
    assert df.pharmpy.column_type[('ID', 'DV')] == \
        [data.ColumnType.UNKNOWN, data.ColumnType.UNKNOWN]

    df.pharmpy.column_type['ID'] = data.ColumnType.ID
    df.pharmpy.column_type['DV'] = data.ColumnType.DV
    with pytest.raises(KeyError):       # Max one id column
        df.pharmpy.column_type['DV'] = data.ColumnType.ID
    with pytest.raises(KeyError):
        df.pharmpy.column_type['NOEXISTS'] = data.ColumnType.COVARIATE
    with pytest.raises(ValueError):
        df.pharmpy.column_type[('ID', 'DV')] = \
            [data.ColumnType.COVARIATE, data.ColumnType.COVARIATE, data.ColumnType.COVARIATE]
    with pytest.raises(KeyError):
        df.pharmpy.column_type['NOAVAIL']
    assert df.pharmpy.column_type['ID'] == data.ColumnType.ID
    assert df.pharmpy.column_type['DV'] == data.ColumnType.DV
    assert df.pharmpy.column_type[['ID', 'DV']] == [data.ColumnType.ID, data.ColumnType.DV]
    df.pharmpy.column_type[['HGT', 'WGT']] = data.ColumnType.COVARIATE
    assert df.pharmpy.labels_by_type[data.ColumnType.COVARIATE] == ['WGT', 'HGT']
    df2 = df.copy()
    assert df2.pharmpy.column_type['ID'] == data.ColumnType.ID
    assert df2.pharmpy.column_type['DV'] == data.ColumnType.DV

    assert df2.pharmpy.labels_by_type[data.ColumnType.ID] == ['ID']
    assert df2.pharmpy.labels_by_type[[data.ColumnType.ID]] == ['ID']
    assert df2.pharmpy.labels_by_type[[]] == []
    assert df2.pharmpy.labels_by_type[data.ColumnType.DV] == ['DV']
    assert df2.pharmpy.labels_by_type[[data.ColumnType.ID, data.ColumnType.DV]] == ['ID', 'DV']

    assert df2.pharmpy.id_label == 'ID'

    df2.pharmpy.column_type[['ID', 'DV']] = (data.ColumnType.COVARIATE, data.ColumnType.IDV)
    assert df2.pharmpy.column_type[['ID', 'DV']] == [data.ColumnType.COVARIATE, data.ColumnType.IDV]
    assert df.pharmpy.column_type['ID'] == data.ColumnType.ID

    df3 = data.PharmDataFrame({1: [1, 1, 2, 2], 2: [0.1, 0.2, 0.5, 0.6],
                               3: [70, 70, 75, 75], 4: [185, 185, 160, 160]})
    assert df3.pharmpy.column_type[1] == data.ColumnType.UNKNOWN
    df3.pharmpy.column_type[[3, 4]] = data.ColumnType.COVARIATE
    df3.pharmpy.column_type[1] = data.ColumnType.ID
    assert df3.pharmpy.column_type[[1, 2, 3, 4]] == \
        [data.ColumnType.ID, data.ColumnType.UNKNOWN,
         data.ColumnType.COVARIATE, data.ColumnType.COVARIATE]


def test_time_varying_covariates(df):
    assert df.pharmpy.time_varying_covariates == ['WGT']
    df_untyped = data.PharmDataFrame({'ID': [1, 1, 2, 2], 'DV': [0.1, 0.2, 0.5, 0.6]})
    assert df_untyped.pharmpy.time_varying_covariates == []


def test_covariate_baselines(df):
    correct_baselines = pd.DataFrame(
        {'WGT': [70, 75], 'HGT': [185, 160]},
        index=pd.Int64Index([1, 2], name='ID')
    )
    pd.testing.assert_frame_equal(df.pharmpy.covariate_baselines, correct_baselines)


def test_write(fs, df):
    df.pharmpy.write_csv(path="my.csv")
    with open("my.csv", "r") as fh:
        contents = fh.read()
    assert contents == "ID,DV,WGT,HGT\n1,0.1,70,185\n1,0.2,72,185\n2,0.5,75,160\n2,0.6,75,160\n"


def test_copy(df):
    new = copy.deepcopy(df)
    assert id(new) != id(df)
