import numpy as np
import pandas as pd
import pytest

from pharmpy.modeling import (
    add_time_after_dose,
    check_dataset,
    deidentify_data,
    drop_columns,
    drop_dropped_columns,
    expand_additional_doses,
    get_cmt,
    get_concentration_parameters_from_data,
    get_covariate_baselines,
    get_doseid,
    get_doses,
    get_evid,
    get_ids,
    get_mdv,
    get_number_of_individuals,
    get_number_of_observations,
    get_number_of_observations_per_individual,
    get_observations,
    list_time_varying_covariates,
    remove_loq_data,
    set_dvid,
    translate_nmtran_time,
    undrop_columns,
)


def test_get_ids(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    assert get_ids(model) == list(range(1, 60))


def test_get_doseid(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    doseid = get_doseid(model)
    assert len(doseid) == 744
    assert doseid[0] == 1
    assert doseid[743] == 13

    # Same timepoint for dose and observation
    df = model.dataset.copy()
    df.loc[742, 'TIME'] = df.loc[743, 'TIME']
    model = model.replace(dataset=df)
    doseid = get_doseid(model)
    assert len(doseid) == 744
    assert doseid[743] == 12
    assert doseid[742] == 13


def test_get_number_of_individuals(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    assert get_number_of_individuals(model) == 59


def test_get_observations(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    ser = get_observations(model)
    assert ser.loc[1, 2.0] == 17.3
    assert ser.loc[2, 63.5] == 24.6
    assert len(ser) == 155


def test_number_of_observations(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    assert get_number_of_observations(model) == 155
    assert list(get_number_of_observations_per_individual(model)) == [
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        3,
        1,
        3,
        2,
        4,
        2,
        3,
        3,
        4,
        3,
        3,
        3,
        2,
        3,
        3,
        6,
        2,
        2,
        1,
        1,
        2,
        1,
        3,
        2,
        2,
        2,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        1,
        3,
        3,
        1,
        1,
        5,
        3,
        4,
        3,
        3,
        2,
        4,
        1,
        1,
        2,
        3,
        3,
    ]


def test_covariate_baselines(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    covs = model.datainfo[['WGT', 'APGR']].set_types('covariate')
    model = model.replace(datainfo=model.datainfo[0:3] + covs + model.datainfo[5:])
    df = get_covariate_baselines(model)
    assert len(df) == 59
    assert list(df.columns) == ['WGT', 'APGR']
    assert df.index.name == 'ID'
    assert df['WGT'].loc[2] == 1.5
    assert df['APGR'].loc[11] == 7.0


def test_doses(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    ser = get_doses(model)
    assert len(ser) == 589
    assert ser.loc[1, 0.0] == 25.0


def test_timevarying_covariates(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    a = list_time_varying_covariates(model)
    assert a == []


def test_get_mdv(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    mdv = get_mdv(model)
    label_test = model.datainfo.typeix['dose'][0].name
    data_test = model.dataset[label_test].astype('float64').squeeze()
    mdv_test = data_test.where(data_test == 0, other=1).astype('int32')
    result = mdv.equals(other=mdv_test)
    assert result is True


def test_get_evid(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    evid = get_evid(model)
    assert evid.sum() == 589


def test_get_cmt(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    cmt = get_cmt(model)
    assert cmt.sum() == 589


def test_add_time_after_dose(load_model_for_test, load_example_model_for_test, testdata):
    m = load_example_model_for_test("pheno")
    m = add_time_after_dose(m)
    tad = m.dataset['TAD']

    assert tad[0] == 0.0
    assert tad[1] == 2.0
    assert tad[743] == 2.0

    m = load_model_for_test(testdata / 'nonmem' / 'models' / 'pef.mod')
    m = add_time_after_dose(m)
    tad = list(m.dataset['TAD'].iloc[0:21])
    assert tad == [
        0.0,
        0.0,
        0.0,
        1.5,
        3.0,
        10.719999999999999,
        0.0,
        0.0,
        0.0,
        1.4500000000000028,
        3.0,
        10.980000000000004,
        0.0,
        0.0,
        2.25,
        3.770000000000003,
        12.0,
        0.0,
        0.0,
        1.4700000000000273,
        2.9700000000000273,
    ]
    assert m.dataset.loc[103, 'TAD'] == 0.0
    assert m.dataset.loc[104, 'TAD'] == pytest.approx(1.17)

    m = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox1.mod')
    m = add_time_after_dose(m)
    tad = list(m.dataset['TAD'].iloc[0:16])
    assert tad == [0.0, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 0.0, 12.0, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0]


def test_get_concentration_parameters_from_data(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    df = get_concentration_parameters_from_data(model)
    assert df['Cmax'].loc[1, 1] == 17.3


def test_drop_columns(load_example_model_for_test):
    m = load_example_model_for_test('pheno')
    m = drop_columns(m, "APGR")
    correct = ['ID', 'TIME', 'AMT', 'WGT', 'DV', 'FA1', 'FA2']
    assert m.datainfo.names == correct
    assert list(m.dataset.columns) == correct
    m = drop_columns(m, ['DV', 'ID'])
    assert m.datainfo.names == ['TIME', 'AMT', 'WGT', 'FA1', 'FA2']
    assert list(m.dataset.columns) == ['TIME', 'AMT', 'WGT', 'FA1', 'FA2']
    m = drop_columns(m, ['TIME'], mark=True)
    assert m.datainfo['TIME'].drop
    assert list(m.dataset.columns) == ['TIME', 'AMT', 'WGT', 'FA1', 'FA2']


def test_drop_dropped_columns(load_example_model_for_test):
    m = load_example_model_for_test('pheno')
    m = drop_dropped_columns(m)
    correct = ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV', 'FA1', 'FA2']
    assert list(m.dataset.columns) == correct
    m = drop_columns(m, ['ID', 'TIME', 'AMT'], mark=True)
    m = drop_dropped_columns(m)
    assert list(m.dataset.columns) == ['WGT', 'APGR', 'DV', 'FA1', 'FA2']


def test_undrop_columns(load_example_model_for_test):
    m = load_example_model_for_test('pheno')
    m = drop_columns(m, ["APGR", "WGT"], mark=True)
    m = undrop_columns(m, "WGT")
    assert not m.datainfo["WGT"].drop
    assert m.datainfo["APGR"].drop


def test_remove_loq_data(load_example_model_for_test):
    m = load_example_model_for_test('pheno')
    m = remove_loq_data(m, lloq=10, uloq=40)
    assert len(m.dataset) == 736


def test_check_dataset(load_example_model_for_test):
    m = load_example_model_for_test('pheno')
    check_dataset(m)

    df = check_dataset(m, verbose=True, dataframe=True)
    assert df is not None
    assert df[df['code'] == 'A1']['result'].iloc[0] == 'OK'
    assert df[df['code'] == 'A4']['result'].iloc[0] == 'SKIP'

    df = m.dataset.copy()
    df.loc[743, 'WGT'] = -1
    m = m.replace(dataset=df)
    df = check_dataset(m, verbose=True, dataframe=True)
    assert df is not None
    assert df[df['code'] == 'A3']['result'].iloc[0] == 'FAIL'


def test_nmtran_time(load_example_model_for_test):
    m = load_example_model_for_test("pheno_linear")
    translate_nmtran_time(m)


def test_expand_additional_doses(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pef.mod')
    model = expand_additional_doses(model)
    df = model.dataset
    assert len(df) == 1494
    assert len(df.columns) == 5
    assert df.loc[0, 'AMT'] == 400.0
    assert df.loc[1, 'AMT'] == 400.0
    assert df.loc[2, 'AMT'] == 400.0
    assert df.loc[3, 'AMT'] == 400.0
    assert df.loc[4, 'AMT'] == 200.0
    assert df.loc[0, 'TIME'] == 0.0
    assert df.loc[1, 'TIME'] == 12.0
    assert df.loc[2, 'TIME'] == 24.0
    assert df.loc[3, 'TIME'] == 36.0
    assert df.loc[4, 'TIME'] == 48.0

    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pef.mod')
    model = expand_additional_doses(model, flag=True)
    df = model.dataset
    assert len(df) == 1494
    assert len(df.columns) == 8
    assert not df.loc[0, 'EXPANDED']
    assert df.loc[1, 'EXPANDED']
    assert df.loc[2, 'EXPANDED']
    assert df.loc[3, 'EXPANDED']
    assert not df.loc[4, 'EXPANDED']


def test_deidentify_data():
    np.random.seed(23)

    example = pd.DataFrame(
        {'ID': [1, 1, 2, 2], 'DATE': ["2012-05-25", "2013-04-02", "2011-12-23", "2005-02-28"]}
    )
    df = deidentify_data(example, date_columns=['DATE'])
    correct = pd.to_datetime(
        pd.Series(["1908-05-25", "1909-04-02", "1907-12-23", "1901-02-28"], name='DATE')
    )
    pd.testing.assert_series_equal(df['DATE'], correct)

    example = pd.DataFrame(
        {
            'ID': [1, 1, 2, 2],
            'DATE': ["2012-05-25", "2013-04-02", "2011-12-23", "2005-02-28"],
            'BIRTH': ["1980-07-07", "1980-07-07", "1956-10-12", "1956-10-12"],
        }
    )
    df = deidentify_data(example, date_columns=['DATE', 'BIRTH'])
    correct_date = pd.to_datetime(
        pd.Series(["1959-12-23", "1953-02-28", "1960-05-25", "1961-04-02"], name='DATE')
    )
    correct_birth = pd.to_datetime(
        pd.Series(["1904-10-12", "1904-10-12", "1928-07-07", "1928-07-07"], name='BIRTH')
    )
    pd.testing.assert_series_equal(df['DATE'], correct_date)
    pd.testing.assert_series_equal(df['BIRTH'], correct_birth)


def test_set_dvid(load_example_model_for_test):
    m = load_example_model_for_test('pheno')
    m = set_dvid(m, 'FA1')
    col = m.datainfo['FA1']
    assert col.type == 'dvid'
    assert col.scale == 'nominal'
    assert col.categories == (0, 1)
    m = set_dvid(m, 'FA1')
    assert m.datainfo['FA1'].type == 'dvid'
    m = set_dvid(m, 'FA2')
    assert m.datainfo['FA1'].type == 'unknown'
    assert m.datainfo['FA2'].type == 'dvid'
    with pytest.raises(ValueError):
        set_dvid(m, 'WGT')
