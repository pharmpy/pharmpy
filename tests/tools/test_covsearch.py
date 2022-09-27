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
        'effects',
        'p_forward',
        'p_backward',
        'max_steps',
        'algorithm',
        'exception',
        'match',
    ),
    [
        (
            None,
            MINIMAL_VALID_MFL_STRING,
            'x',
            0.01,
            -1,
            'scm-forward-then-backward',
            TypeError,
            'Invalid `p_forward`',
        ),
        (
            None,
            MINIMAL_VALID_MFL_STRING,
            1.05,
            0.01,
            -1,
            'scm-forward-then-backward',
            ValueError,
            'Invalid `p_forward`',
        ),
        (
            None,
            MINIMAL_VALID_MFL_STRING,
            0.05,
            [],
            -1,
            'scm-forward-then-backward',
            TypeError,
            'Invalid `p_backward`',
        ),
        (
            None,
            MINIMAL_VALID_MFL_STRING,
            0.05,
            1.01,
            -1,
            'scm-forward-then-backward',
            ValueError,
            'Invalid `p_backward`',
        ),
        (
            None,
            MINIMAL_VALID_MFL_STRING,
            0.05,
            0.01,
            1.2,
            'scm-forward-then-backward',
            TypeError,
            'Invalid `max_steps`',
        ),
        (None, MINIMAL_VALID_MFL_STRING, 0.05, 0.01, -1, (), TypeError, 'Invalid `algorithm`'),
        (
            None,
            MINIMAL_VALID_MFL_STRING,
            0.05,
            0.01,
            -1,
            'scm-backward',
            ValueError,
            'Invalid `algorithm`',
        ),
        (('nonmem', 'pheno.mod'), 1, 0.05, 0.01, -1, 'scm-forward', TypeError, 'Invalid `effects`'),
        (
            ('nonmem', 'pheno.mod'),
            MINIMAL_INVALID_MFL_STRING,
            0.05,
            0.01,
            -1,
            'scm-forward',
            ValueError,
            'Invalid `effects`',
        ),
        (
            ('nonmem', 'pheno.mod'),
            (
                ('CL', 'WGT', 'exp', '*'),
                ('VC', 'WGT', 'exp', '*'),
            ),
            0.05,
            0.01,
            -1,
            'scm-forward',
            ValueError,
            'Invalid `effects` because of invalid parameter',
        ),
        (
            ('nonmem', 'pheno.mod'),
            (
                ('CL', 'WGT', 'exp', '*'),
                ('V', 'SEX', 'exp', '*'),
            ),
            0.05,
            0.01,
            -1,
            'scm-forward',
            ValueError,
            'Invalid `effects` because of invalid covariate',
        ),
        (
            ('nonmem', 'pheno.mod'),
            (
                ('CL', 'WGT', 'exp', '*'),
                ('V', 'WGT', 'abc', '*'),
            ),
            0.05,
            0.01,
            -1,
            'scm-forward',
            ValueError,
            'Invalid `effects` because of invalid effect function',
        ),
        (
            ('nonmem', 'pheno.mod'),
            (
                ('CL', 'WGT', 'exp', '*'),
                ('V', 'WGT', 'exp', '-'),
            ),
            0.05,
            0.01,
            -1,
            'scm-forward',
            ValueError,
            'Invalid `effects` because of invalid effect operation',
        ),
    ],
)
def test_validate_input_raises(
    load_model_for_test,
    testdata,
    model_path,
    effects,
    p_forward,
    p_backward,
    max_steps,
    algorithm,
    exception,
    match,
):

    model = load_model_for_test(testdata.joinpath(*model_path)) if model_path else None

    with pytest.raises(exception, match=match):
        validate_input(
            effects,
            p_forward=p_forward,
            p_backward=p_backward,
            max_steps=max_steps,
            algorithm=algorithm,
            model=model,
        )


def test_validate_input_raises_on_wrong_model_type():
    with pytest.raises(TypeError, match='Invalid `model`'):
        validate_input(MINIMAL_VALID_MFL_STRING, model=1)
