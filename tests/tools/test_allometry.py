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
        'allometric_variable',
        'reference_value',
        'parameters',
        'initials',
        'lower_bounds',
        'upper_bounds',
        'fixed',
    ),
    [
        (
            None,
            1,
            70,
            None,
            None,
            None,
            None,
            True,
        ),
        (
            None,
            'WT',
            [],
            None,
            None,
            None,
            None,
            True,
        ),
        (
            None,
            'WT',
            70,
            'CL',
            None,
            None,
            None,
            True,
        ),
        (
            None,
            'WT',
            70,
            None,
            0.1,
            None,
            None,
            True,
        ),
        (
            None,
            'WT',
            70,
            None,
            None,
            0.0001,
            None,
            True,
        ),
        (
            None,
            'WT',
            70,
            None,
            None,
            None,
            1000,
            True,
        ),
        (
            None,
            'WT',
            70,
            None,
            None,
            None,
            None,
            1,
        ),
        (
            ('nonmem', 'pheno.mod'),
            'WT',
            70,
            None,
            None,
            None,
            None,
            True,
        ),
        (
            ('nonmem', 'pheno.mod'),
            'WGT',
            70,
            ['K'],
            None,
            None,
            None,
            True,
        ),
    ],
)
def test_validate_input_raises(
    load_model_for_test,
    testdata,
    model_path,
    allometric_variable,
    reference_value,
    parameters,
    initials,
    lower_bounds,
    upper_bounds,
    fixed,
):

    model = load_model_for_test(testdata.joinpath(*model_path)) if model_path else None

    with pytest.raises((ValueError, TypeError)):
        validate_input(
            allometric_variable=allometric_variable,
            reference_value=reference_value,
            parameters=parameters,
            initials=initials,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            fixed=fixed,
            model=model,
        )


def test_validate_input_raises_on_wrong_model_type():
    with pytest.raises(TypeError, match='Invalid model'):
        validate_input(model=1)
