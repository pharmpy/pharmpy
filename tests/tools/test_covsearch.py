import pytest

from pharmpy.tools.covsearch.tool import create_workflow, validate_input
from pharmpy.workflows import Workflow

MINIMAL_INVALID_MFL_STRING = ''
MINIMAL_VALID_MFL_STRING = 'LET(x, 0)'
LARGE_VALID_MFL_STRING = 'COVARIATE(@IIV, @CONTINUOUS, *);COVARIATE(@IIV, @CATEGORICAL, CAT)'


def test_create_workflow():
    assert isinstance(create_workflow(MINIMAL_VALID_MFL_STRING), Workflow)


def test_create_workflow_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    assert isinstance(create_workflow(MINIMAL_VALID_MFL_STRING, model=model), Workflow)


def test_validate_input():
    validate_input(MINIMAL_VALID_MFL_STRING)


@pytest.mark.parametrize(
    ('model_path',), ((('nonmem', 'pheno.mod'),), (('nonmem', 'ruvsearch', 'mox3.mod'),))
)
def test_validate_input_with_model(load_model_for_test, testdata, model_path):
    model = load_model_for_test(testdata.joinpath(*model_path))
    validate_input(LARGE_VALID_MFL_STRING, model=model)


@pytest.mark.parametrize(
    (
        'model_path',
        'arguments',
        'exception',
        'match',
    ),
    [
        (
            None,
            dict(p_forward='x'),
            TypeError,
            'Invalid `p_forward`',
        ),
        (
            None,
            dict(p_forward=1.05),
            ValueError,
            'Invalid `p_forward`',
        ),
        (
            None,
            dict(p_backward=[]),
            TypeError,
            'Invalid `p_backward`',
        ),
        (
            None,
            dict(p_backward=1.01),
            ValueError,
            'Invalid `p_backward`',
        ),
        (
            None,
            dict(max_steps=1.2),
            TypeError,
            'Invalid `max_steps`',
        ),
        (None, dict(algorithm=()), TypeError, 'Invalid `algorithm`'),
        (
            None,
            dict(algorithm='scm-backward'),
            ValueError,
            'Invalid `algorithm`',
        ),
        (('nonmem', 'pheno.mod'), dict(effects=1), TypeError, 'Invalid `effects`'),
        (
            ('nonmem', 'pheno.mod'),
            dict(effects=MINIMAL_INVALID_MFL_STRING),
            ValueError,
            'Invalid `effects`',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(effects='LAGTIME()'),
            ValueError,
            'Invalid `effects`',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(effects='COVARIATE([CL, VC], WGT, EXP)'),
            ValueError,
            'Invalid `effects` because of invalid parameter',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(effects='COVARIATE([CL, V], SEX, EXP)'),
            ValueError,
            'Invalid `effects` because of invalid covariate',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(effects='COVARIATE([CL, V], WGT, [EXP, ABC])'),
            ValueError,
            'Invalid `effects` because of invalid effect function',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(
                effects=(
                    ('CL', 'WGT', 'exp', '*'),
                    ('V', 'WGT', 'exp', '-'),
                )
            ),
            ValueError,
            'Invalid `effects`',
        ),
        (
            None,
            dict(model=1),
            TypeError,
            'Invalid `model`',
        ),
    ],
)
def test_validate_input_raises(
    load_model_for_test,
    testdata,
    model_path,
    arguments,
    exception,
    match,
):
    model = load_model_for_test(testdata.joinpath(*model_path)) if model_path else None

    harmless_arguments = dict(
        effects=MINIMAL_VALID_MFL_STRING,
    )

    kwargs = {**harmless_arguments, 'model': model, **arguments}

    with pytest.raises(exception, match=match):
        validate_input(**kwargs)
