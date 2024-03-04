import pytest

from pharmpy.tools.allometry.tool import create_workflow, validate_input
from pharmpy.workflows import Workflow


def test_create_workflow():
    assert isinstance(create_workflow(), Workflow)


def test_create_workflow_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert isinstance(create_workflow(model=model, allometric_variable='WGT'), Workflow)


def test_validate_input():
    validate_input()


def test_validate_input_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    validate_input(model=model, allometric_variable='WGT')


def test_validate_input_with_model_and_parameters(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    validate_input(model=model, allometric_variable='WGT', parameters=['CL', 'V'])


@pytest.mark.parametrize(
    (
        'model_path',
        'arguments',
        'exception',
        'match',
    ),
    [
        (None, dict(allometric_variable=1), TypeError, 'Invalid `allometric_variable`'),
        (None, dict(reference_value=[]), TypeError, 'Invalid `reference_value`'),
        (None, dict(parameters='CL'), TypeError, 'Invalid `parameters`'),
        (None, dict(initials=0.1), TypeError, 'Invalid `initials`'),
        (None, dict(lower_bounds=0.0001), TypeError, 'Invalid `lower_bounds`'),
        (None, dict(upper_bounds=1000), TypeError, 'Invalid `upper_bounds`'),
        (None, dict(fixed=1), TypeError, 'Invalid `fixed`'),
        (
            ('nonmem', 'pheno.mod'),
            dict(allometric_variable='WT'),
            ValueError,
            'Invalid `allometric_variable`',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(allometric_variable='WGT', parameters=['K']),
            ValueError,
            'Invalid `parameters`',
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

    kwargs = {'model': model, **arguments}

    with pytest.raises(exception, match=match):
        validate_input(**kwargs)
