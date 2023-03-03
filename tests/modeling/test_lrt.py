import numpy as np
import pytest

from pharmpy.modeling import add_allometry, add_covariate_effect
from pharmpy.modeling.lrt import best_of_many, best_of_two, cutoff, degrees_of_freedom, p_value
from pharmpy.modeling.lrt import test as lrt_test


@pytest.mark.parametrize(
    ('model_path', 'effects', 'expected', 'allow_nested'),
    [
        (
            ('nonmem', 'pheno.mod'),
            [],
            0,
            False,
        ),
        (
            ('nonmem', 'pheno.mod'),
            [('CL', 'WGT', 'exp', '*')],
            1,
            False,
        ),
        (
            ('nonmem', 'pheno.mod'),
            [('CL', 'APGR', 'cat', '*')],
            9,
            False,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'piece_lin', '*')],
            2,
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'theta1 * (cov/median)**theta2', '*')],
            2,
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', '((cov/std) - median) * theta', '*')],
            1,
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [
                ('CL', 'WGT', 'exp', '+'),
                ('V', 'WGT', 'exp', '+'),
            ],
            2,
            True,
        ),
    ],
    ids=repr,
)
def test_degrees_of_freedom(
    load_model_for_test, testdata, model_path, effects, expected, allow_nested
):
    parent = load_model_for_test(testdata.joinpath(*model_path))
    child = parent

    for effect in effects:
        child = add_covariate_effect(child, *effect, allow_nested=allow_nested)

    assert degrees_of_freedom(parent, child) == expected


def test_cutoff(load_model_for_test, testdata):
    parent = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert cutoff(parent, parent, 0.05) == 0

    child = add_allometry(
        parent,
        allometric_variable='WGT',
        reference_value=70,
        parameters=['CL'],
        initials=[0.7],
        lower_bounds=[0],
        upper_bounds=[2],
        fixed=True,
    )

    assert cutoff(parent, child, 0.05) == pytest.approx(3.8414588206941285)
    assert cutoff(child, parent, 0.05) == -cutoff(parent, child, 0.05)


def test_p_value(load_model_for_test, testdata):
    parent = load_model_for_test(testdata / 'nonmem' / 'pheno_abbr.mod')
    child = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')

    alpha = 0.05
    parent_ofv = 10
    child_ofv = parent_ofv - cutoff(parent, child, alpha)

    assert p_value(parent, child, parent_ofv, child_ofv) == pytest.approx(alpha)


def test_test(load_model_for_test, testdata):
    parent = load_model_for_test(testdata / 'nonmem' / 'pheno_abbr.mod')
    child = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')

    alpha = 0.05
    parent_ofv = 10
    child_ofv = parent_ofv - cutoff(parent, child, alpha) - 0.01

    assert lrt_test(parent, child, parent_ofv, child_ofv, alpha)


def test_best_of_two(load_model_for_test, testdata):
    parent = load_model_for_test(testdata / 'nonmem' / 'pheno_abbr.mod')
    child = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')

    alpha = 0.05
    parent_ofv = 10
    child_ofv = parent_ofv - cutoff(parent, child, alpha) - 0.01

    assert best_of_two(parent, child, parent_ofv, child_ofv, alpha) is child


def test_best_of_many(load_model_for_test, testdata):
    parent = load_model_for_test(testdata / 'nonmem' / 'pheno_abbr.mod')
    child = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')

    alpha = 0.05
    parent_ofv = 10
    child_ofv = parent_ofv - cutoff(parent, child, alpha) - 0.01

    assert (
        best_of_many(parent, [parent, child], parent_ofv, [parent_ofv, child_ofv], alpha) is child
    )


def test_best_of_many_nan(load_model_for_test, testdata):
    parent = load_model_for_test(testdata / 'nonmem' / 'pheno_abbr.mod')
    child = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')

    alpha = 0.05
    parent_ofv = 10
    child_ofv = parent_ofv - cutoff(parent, child, alpha) - 0.01

    assert (
        best_of_many(parent, [child, child, child], parent_ofv, [np.nan, np.nan, np.nan], alpha)
        is parent
    )
    assert (
        best_of_many(parent, [child, child, child], parent_ofv, [np.nan, child_ofv, np.nan], alpha)
        is child
    )
    assert (
        best_of_many(
            parent, [child, child, child], parent_ofv, [np.nan, child_ofv + 0.02, np.nan], alpha
        )
        is parent
    )
