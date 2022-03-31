from pharmpy.modeling import (
    add_time_after_dose,
    check_dataset,
    drop_columns,
    drop_dropped_columns,
    expand_additional_doses,
    get_concentration_parameters_from_data,
    get_covariate_baselines,
    get_doseid,
    get_doses,
    get_ids,
    get_mdv,
    get_number_of_individuals,
    get_number_of_observations,
    get_number_of_observations_per_individual,
    get_observations,
    list_time_varying_covariates,
    load_example_model,
    read_model,
    remove_loq_data,
    translate_nmtran_time,
    undrop_columns,
)

model = load_example_model("pheno")


def test_get_ids():
    assert get_ids(model) == list(range(1, 60))


def test_get_doseid():
    doseid = get_doseid(model)
    assert len(doseid) == 744
    assert doseid[0] == 1
    assert doseid[743] == 13

    # Same timepoint for dose and observation
    newmod = model.copy()
    newmod.dataset.loc[742, 'TIME'] = newmod.dataset.loc[743, 'TIME']
    doseid = get_doseid(newmod)
    assert len(doseid) == 744
    assert doseid[743] == 12
    assert doseid[742] == 13


def test_get_number_of_individuals():
    assert get_number_of_individuals(model) == 59


def test_get_observations():
    ser = get_observations(model)
    assert ser.loc[1, 2.0] == 17.3
    assert ser.loc[2, 63.5] == 24.6
    assert len(ser) == 155


def test_number_of_observations():
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


def test_covariate_baselines():
    model.datainfo[['WGT', 'APGR']].types = 'covariate'
    df = get_covariate_baselines(model)
    assert len(df) == 59
    assert list(df.columns) == ['WGT', 'APGR']
    assert df.index.name == 'ID'
    assert df['WGT'].loc[2] == 1.5
    assert df['APGR'].loc[11] == 7.0


def test_doses():
    ser = get_doses(model)
    assert len(ser) == 589
    assert ser.loc[1, 0.0] == 25.0


def test_timevarying_covariates():
    a = list_time_varying_covariates(model)
    assert a == []


def test_get_mdv():
    mdv = get_mdv(model)
    label_test = model.datainfo.typeix['dose'][0].name
    data_test = model.dataset[label_test].astype('float64').squeeze()
    mdv_test = data_test.where(data_test == 0, other=1).astype('int64')
    result = mdv.equals(other=mdv_test)
    assert result is True


def test_add_time_after_dose(testdata):
    model = load_example_model("pheno")
    m = model.copy()
    add_time_after_dose(m)
    tad = m.dataset['TAD']
    assert tad[0] == 0.0
    assert tad[1] == 2.0
    assert tad[743] == 2.0

    m = read_model(testdata / 'nonmem' / 'models' / 'pef.mod')
    add_time_after_dose(m)
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

    m = read_model(testdata / 'nonmem' / 'models' / 'mox1.mod')
    add_time_after_dose(m)
    tad = list(m.dataset['TAD'].iloc[0:16])
    assert tad == [0.0, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 0.0, 12.0, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0]


def test_get_concentration_parameters_from_data():
    df = get_concentration_parameters_from_data(model)
    assert df['Cmax'].loc[1, 1] == 17.3


def test_drop_columns():
    m = model.copy()
    drop_columns(m, "APGR")
    correct = ['ID', 'TIME', 'AMT', 'WGT', 'DV']
    assert m.datainfo.names == correct
    assert list(m.dataset.columns) == correct + ['FA1', 'FA2']
    drop_columns(m, ['DV', 'ID'])
    assert m.datainfo.names == ['TIME', 'AMT', 'WGT']
    assert list(m.dataset.columns) == ['TIME', 'AMT', 'WGT', 'FA1', 'FA2']
    drop_columns(m, ['TIME'], mark=True)
    assert m.datainfo['TIME'].drop
    assert list(m.dataset.columns) == ['TIME', 'AMT', 'WGT', 'FA1', 'FA2']


def test_drop_dropped_columns():
    m = model.copy()
    drop_dropped_columns(m)
    correct = ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV']
    assert list(m.dataset.columns) == correct
    drop_columns(m, ['ID', 'TIME', 'AMT'], mark=True)
    drop_dropped_columns(m)
    assert list(m.dataset.columns) == ['WGT', 'APGR', 'DV']


def test_undrop_columns():
    m = model.copy()
    drop_columns(m, ["APGR", "WGT"], mark=True)
    undrop_columns(m, "WGT")
    assert not m.datainfo["WGT"].drop
    assert m.datainfo["APGR"].drop


def test_remove_loq_data():
    m = model.copy()
    remove_loq_data(m, lloq=10, uloq=40)
    assert len(m.dataset) == 736


def test_check_dataset():
    m = model.copy()
    check_dataset(m)

    df = check_dataset(m, verbose=True, dataframe=True)
    assert df[df['code'] == 'A1']['result'].iloc[0] == 'OK'
    assert df[df['code'] == 'A4']['result'].iloc[0] == 'SKIP'

    m.dataset.loc[743, 'WGT'] = -1
    df = check_dataset(m, verbose=True, dataframe=True)
    assert df[df['code'] == 'A3']['result'].iloc[0] == 'FAIL'


def test_nmtran_time():
    m = load_example_model("pheno_linear")
    translate_nmtran_time(m)


def test_expand_additional_doses(testdata):
    model = read_model(testdata / 'nonmem' / 'models' / 'pef.mod')
    expand_additional_doses(model)
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

    model = read_model(testdata / 'nonmem' / 'models' / 'pef.mod')
    expand_additional_doses(model, flag=True)
    df = model.dataset
    assert len(df) == 1494
    assert len(df.columns) == 8
    assert not df.loc[0, 'EXPANDED']
    assert df.loc[1, 'EXPANDED']
    assert df.loc[2, 'EXPANDED']
    assert df.loc[3, 'EXPANDED']
    assert not df.loc[4, 'EXPANDED']
