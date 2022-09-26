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
    ('model_path', 'effects', 'p_forward', 'p_backward', 'max_steps', 'algorithm'),
    [
        (
            None,
            MINIMAL_VALID_MFL_STRING,
            1.05,
            0.01,
            -1,
            'scm-forward-then-backward',
        ),
        (
            None,
            MINIMAL_VALID_MFL_STRING,
            0.05,
            1.01,
            -1,
            'scm-forward-then-backward',
        ),
        (
            None,
            MINIMAL_VALID_MFL_STRING,
            0.05,
            0.01,
            1.2,
            'scm-forward-then-backward',
        ),
        (
            None,
            MINIMAL_VALID_MFL_STRING,
            0.05,
            0.01,
            -1,
            'scm-backward',
        ),
        (
            ('nonmem', 'pheno.mod'),
            MINIMAL_INVALID_MFL_STRING,
            0.05,
            0.01,
            -1,
            'scm-forward',
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
):

    model = load_model_for_test(testdata.joinpath(*model_path)) if model_path else None

    with pytest.raises((ValueError, TypeError)):
        validate_input(
            effects,
            p_forward=p_forward,
            p_backward=p_backward,
            max_steps=max_steps,
            algorithm=algorithm,
            model=model,
        )


def test_validate_input_raises_on_wrong_model_type():
    with pytest.raises(TypeError, match='Invalid model'):
        validate_input(MINIMAL_VALID_MFL_STRING, model=1)
