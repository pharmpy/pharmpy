from pharmpy import Model
from pharmpy.modeling import (
    get_number_of_individuals,
    get_number_of_observations,
    get_number_of_observations_per_individual,
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
