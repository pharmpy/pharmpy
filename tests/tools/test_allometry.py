import pytest

from pharmpy.tools.allometry.tool import create_workflow
from pharmpy.workflows import Workflow


def test_create_workflow():
    assert isinstance(create_workflow(), Workflow)


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
    ],
)
def test_create_workflow_raises(
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
        create_workflow(
            allometric_variable=allometric_variable,
            reference_value=reference_value,
            parameters=parameters,
            initials=initials,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            fixed=fixed,
            model=model,
        )
