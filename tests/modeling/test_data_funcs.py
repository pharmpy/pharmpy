from pharmpy.modeling import (
    add_time_after_dose,
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
)

model = load_example_model("pheno")


def test_get_ids():
    assert get_ids(model) == list(range(1, 60))


def test_get_doseid():
    doseid = get_doseid(model)
    assert len(doseid) == 744
    assert doseid[0] == 1
    assert doseid[743] == 13


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


def test_add_time_after_dose():
    m = model.copy()
    add_time_after_dose(m)
    tad = m.dataset['TAD']
    assert tad[0] == 0.0
    assert tad[1] == 2.0
    assert tad[743] == 2.0


def test_get_concentration_parameters_from_data():
    df = get_concentration_parameters_from_data(model)
    assert df['Cmax'].loc[1, 1] == 17.3
