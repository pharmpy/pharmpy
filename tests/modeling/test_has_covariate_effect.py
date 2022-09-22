import pytest

from pharmpy.modeling import has_covariate_effect


@pytest.mark.parametrize(
    ('model_path', 'effect', 'has'),
    [
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('CL', 'PREP'),
            True,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('V1', 'PREP'),
            True,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('V2', 'PREP'),
            False,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('Q', 'PREP'),
            False,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('CL', 'AGE'),
            True,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('V1', 'AGE'),
            False,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('V2', 'AGE'),
            False,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('Q', 'AGE'),
            False,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('CL', 'OCC'),
            True,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('V1', 'OCC'),
            True,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('V2', 'OCC'),
            False,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('Q', 'OCC'),
            False,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            ('CL', 'WGT'),
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            ('V', 'WGT'),
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            ('CL', 'APGR'),
            False,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            ('V', 'APGR'),
            True,
        ),
        (
            ('nonmem', 'pheno.mod'),
            ('CL', 'WGT'),
            False,
        ),
        (
            ('nonmem', 'pheno.mod'),
            ('V', 'WGT'),
            False,
        ),
        (
            ('nonmem', 'pheno.mod'),
            ('CL', 'APGR'),
            False,
        ),
        (
            ('nonmem', 'pheno.mod'),
            ('V', 'APGR'),
            False,
        ),
    ],
    ids=repr,
)
def test_has_covariate_effect(load_model_for_test, testdata, model_path, effect, has):

    model = load_model_for_test(testdata.joinpath(*model_path))

    assert has_covariate_effect(model, *effect) is has
