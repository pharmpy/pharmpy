from io import StringIO

import pytest

from pharmpy import Model
from pharmpy.modeling import (
    add_covariance_step,
    add_estimation_step,
    append_estimation_step_options,
    generate_model_code,
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
def test_set_estimation_step(testdata, method, kwargs, code_ref):
    model = Model.create_model(testdata / 'nonmem' / 'minimal.mod')
    set_estimation_step(model, method, **kwargs)
    assert generate_model_code(model).split('\n')[-2] == code_ref


def test_set_estimation_step_est_middle(testdata):
    model = Model.create_model(
        StringIO(
            '''$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
Y = THETA(1) + ETA(1) + ERR(1)

$ESTIMATION METHOD=COND INTERACTION MAXEVAL=999999
$THETA 0.1
$OMEGA 0.01
$SIGMA 1
'''
        )
    )
    set_estimation_step(model, 'FOCE', interaction=True, cov=True, idx=0)
    assert '$ESTIMATION METHOD=COND INTER MAXEVAL=999999\n$COVARIANCE' in model.model_code


def test_add_estimation_step(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    add_estimation_step(model, 'fo')
    assert len(model.estimation_steps) == 2
    assert generate_model_code(model).split('\n')[-2] == '$ESTIMATION METHOD=ZERO'

    model = Model.create_model(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    add_estimation_step(model, 'fo', evaluation=True)
    assert len(model.estimation_steps) == 2
    assert generate_model_code(model).split('\n')[-2] == '$ESTIMATION METHOD=ZERO MAXEVAL=0'


def test_add_estimation_step_non_int(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'minimal.mod')
    add_estimation_step(model, 'fo', idx=1.0)
    with pytest.raises(TypeError, match='Index must be integer'):
        add_estimation_step(model, 'fo', idx=1.5)


def test_remove_estimation_step(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    remove_estimation_step(model, 0)
    assert not model.estimation_steps
    assert generate_model_code(model).split('\n')[-2] == '$SIGMA 1'


def test_add_covariance_step(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    add_covariance_step(model)
    print(model.model_code)
    assert len(model.estimation_steps) == 1
    assert generate_model_code(model).split('\n')[-2] == '$COVARIANCE'


def test_remove_covariance_step(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'minimal.mod')
    add_covariance_step(model)
    assert generate_model_code(model).split('\n')[-2] == '$COVARIANCE'
    remove_covariance_step(model)
    assert (
        generate_model_code(model).split('\n')[-2]
        == '$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC'
    )


def test_append_estimation_step_options(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    append_estimation_step_options(model, {'SADDLE_RESET': 1}, 0)
    assert (
        model.model_code.split('\n')[-2]
        == '$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC SADDLE_RESET=1'
    )


def test_set_evaluation_step(testdata):
    model = Model.create_model(
        StringIO(
            '''$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
Y = THETA(1) + ETA(1) + ERR(1)

$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$ESTIMATION METHOD=COND INTERACTION MAXEVAL=999999
'''
        )
    )
    set_evaluation_step(model)
    assert model.model_code.split('\n')[-2] == '$ESTIMATION METHOD=COND INTER MAXEVAL=0'
