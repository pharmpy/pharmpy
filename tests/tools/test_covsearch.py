import pytest

from pharmpy.tools.covsearch.tool import create_workflow, validate_input
from pharmpy.workflows import Workflow


def test_create_workflow():
    assert isinstance(create_workflow(''), Workflow)


@pytest.mark.parametrize(
    ('model_path', 'effects', 'p_forward', 'p_backward', 'max_steps', 'algorithm'),
    [
        (
            None,
            '',
            1.05,
            0.01,
            -1,
            'scm-forward-then-backward',
        ),
        (
            None,
            '',
            0.05,
            1.01,
            -1,
            'scm-forward-then-backward',
        ),
        (
            None,
            '',
            0.05,
            0.01,
            1.2,
            'scm-forward-then-backward',
        ),
        (
            None,
            '',
            0.05,
            0.01,
            -1,
            'scm-backward',
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
