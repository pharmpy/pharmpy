import pytest

from pharmpy.modeling import (
    add_covariance_step,
    add_estimation_step,
    append_estimation_step_options,
    remove_covariance_step,
    remove_estimation_step,
    set_estimation_step,
    set_evaluation_step,
)


@pytest.mark.parametrize(
    'method,kwargs,code_ref',
    [
        (
            'fo',
            {'interaction': False},
            '$ESTIMATION METHOD=ZERO MAXEVAL=9990 PRINT=2 POSTHOC',
        ),
        (
            'fo',
            {'interaction': True},
            '$ESTIMATION METHOD=ZERO INTER MAXEVAL=9990 PRINT=2 POSTHOC',
        ),
        (
            'fo',
            {'tool_options': {'saddle_reset': 1}},
            '$ESTIMATION METHOD=ZERO INTER MAXEVAL=9990 PRINT=2 SADDLE_RESET=1',
        ),
        (
            'bayes',
            {'interaction': True},
            '$ESTIMATION METHOD=BAYES INTER MAXEVAL=9990 PRINT=2 POSTHOC',
        ),
        (
            'fo',
            {'interaction': False, 'evaluation': True, 'maximum_evaluations': None},
            '$ESTIMATION METHOD=ZERO MAXEVAL=0 PRINT=2 POSTHOC',
        ),
    ],
)
def test_set_estimation_step(testdata, load_model_for_test, method, kwargs, code_ref):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = set_estimation_step(model, method, **kwargs)
    assert model.model_code.split('\n')[-2] == code_ref


def test_set_estimation_step_est_middle(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = set_estimation_step(model, 'FOCE', interaction=True, cov='SANDWICH', idx=0)
    assert (
        '$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC\n$COVARIANCE'
        in model.model_code
    )


def test_add_estimation_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    model = add_estimation_step(model, 'fo')
    assert len(model.estimation_steps) == 2
    assert model.model_code.split('\n')[-2] == '$ESTIMATION METHOD=ZERO'

    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    model = add_estimation_step(model, 'fo', evaluation=True)
    assert len(model.estimation_steps) == 2
    assert model.model_code.split('\n')[-2] == '$ESTIMATION METHOD=ZERO MAXEVAL=0'


def test_add_estimation_step_non_int(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = add_estimation_step(model, 'fo', idx=1.0)
    with pytest.raises(TypeError, match='Index must be integer'):
        add_estimation_step(model, 'fo', idx=1.5)


def test_remove_estimation_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    model = remove_estimation_step(model, 0)
    assert not model.estimation_steps
    assert model.model_code.split('\n')[-2] == '$SIGMA 1'


def test_add_covariance_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    model = add_covariance_step(model, 'SANDWICH')
    assert len(model.estimation_steps) == 1
    assert model.model_code.split('\n')[-2] == '$COVARIANCE'
    model = remove_covariance_step(model)
    model = add_covariance_step(model, 'CPG')
    assert len(model.estimation_steps) == 1
    assert model.model_code.split('\n')[-2] == '$COVARIANCE MATRIX=S'


def test_remove_covariance_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = add_covariance_step(model, 'SANDWICH')
    assert model.model_code.split('\n')[-2] == '$COVARIANCE'
    model = remove_covariance_step(model)
    assert (
        model.model_code.split('\n')[-2]
        == '$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC'
    )


def test_append_estimation_step_options(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    model = append_estimation_step_options(model, {'SADDLE_RESET': 1}, 0)
    assert (
        model.model_code.split('\n')[-2]
        == '$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC SADDLE_RESET=1'
    )


def test_set_evaluation_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = set_evaluation_step(model)
    assert (
        model.model_code.split('\n')[-2]
        == '$ESTIMATION METHOD=COND INTER MAXEVAL=0 PRINT=2 POSTHOC'
    )
