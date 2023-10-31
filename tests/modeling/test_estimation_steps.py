import pytest

from pharmpy.model import SimulationStep
from pharmpy.modeling import (
    add_estimation_step,
    add_parameter_uncertainty_step,
    append_estimation_step_options,
    remove_estimation_step,
    remove_parameter_uncertainty_step,
    set_estimation_step,
    set_evaluation_step,
    set_simulation,
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
    model = set_estimation_step(
        model, 'FOCE', interaction=True, parameter_uncertainty_method='SANDWICH', idx=0
    )
    assert (
        '$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC\n$COVARIANCE UNCONDITIONAL PRINT=E PRECOND=1'
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


def test_add_parameter_uncertainty_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    model = add_parameter_uncertainty_step(model, 'SANDWICH')
    assert len(model.estimation_steps) == 1
    assert model.model_code.split('\n')[-2] == '$COVARIANCE UNCONDITIONAL PRINT=E PRECOND=1'
    model = remove_parameter_uncertainty_step(model)
    model = add_parameter_uncertainty_step(model, 'CPG')
    assert len(model.estimation_steps) == 1
    assert (
        model.model_code.split('\n')[-2] == '$COVARIANCE MATRIX=S UNCONDITIONAL PRINT=E PRECOND=1'
    )

    model = remove_parameter_uncertainty_step(model)

    model = add_parameter_uncertainty_step(model, "EFIM")
    assert len(model.estimation_steps) == 1
    assert (
        "$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC MSFO=efim.msf\n"
        "$PROBLEM DESIGN\n"
        "$DATA file.csv IGNORE=@ REWIND\n"
        "$INPUT ID DV TIME\n"
        "$MSFI efim.msf\n"
        "$DESIGN APPROX=FO FIMDIAG=1 GROUPSIZE=1 OFVTYPE=1\n" in model.model_code
    )


def test_remove_parameter_uncertainty_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = add_parameter_uncertainty_step(model, 'SANDWICH')
    assert model.model_code.split('\n')[-2] == '$COVARIANCE UNCONDITIONAL PRINT=E PRECOND=1'
    model = remove_parameter_uncertainty_step(model)
    assert (
        model.model_code.split('\n')[-2]
        == '$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC'
    )

    model = add_parameter_uncertainty_step(model, "EFIM")
    assert len(model.estimation_steps) == 1
    assert (
        "$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC MSFO=efim.msf\n"
        "$PROBLEM DESIGN\n"
        "$DATA file.csv IGNORE=@ REWIND\n"
        "$INPUT ID DV TIME\n"
        "$MSFI efim.msf\n"
        "$DESIGN APPROX=FO FIMDIAG=1 GROUPSIZE=1 OFVTYPE=1\n" in model.model_code
    )
    model = remove_parameter_uncertainty_step(model)
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


def test_set_simulation(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = set_simulation(model, n=2, seed=1234)
    assert len(model.estimation_steps) == 1
    assert model.estimation_steps[0] == SimulationStep(n=2, seed=1234)
    assert model.model_code.split('\n')[-2] == "$SIMULATION (1234) SUBPROBLEMS=2"
