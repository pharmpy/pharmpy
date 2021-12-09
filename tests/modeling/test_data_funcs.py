from pharmpy import Model
from pharmpy.data import ColumnType
from pharmpy.modeling import (
    get_mdv,
    get_number_of_individuals,
    get_number_of_observations,
    get_number_of_observations_per_individual,
    get_observations,
)


def test_get_number_of_individuals(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    assert get_number_of_individuals(model) == 59


def test_number_of_observations(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
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


def test_get_observations(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    obs = get_observations(model)
    assert len(obs) == 155
    assert obs.loc[1, 2.0] == 17.3


def test_get_mdv(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    mdv = get_mdv(model)
    label_test = model.dataset.pharmpy.labels_by_type[ColumnType.DOSE]
    dose_test = model.dataset[label_test].astype('float64').squeeze()
    mdv_test = dose_test.where(dose_test == 0, other=1).astype('int64')
    result = mdv.equals(other=mdv_test)
    assert result is True
