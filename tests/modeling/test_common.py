from pathlib import Path

import pytest
import sympy
from pyfakefs.fake_filesystem_unittest import Patcher

from pharmpy import Model
from pharmpy.modeling import (
    add_estimation_step,
    append_estimation_step_options,
    convert_model,
    copy_model,
    fix_parameters,
    fix_parameters_to,
    generate_model_code,
    get_model_covariates,
    load_example_model,
    read_model,
    read_model_from_string,
    remove_estimation_step,
    set_estimation_step,
    set_name,
    unfix_parameters,
    unfix_parameters_to,
    write_model,
)


def test_read_model(testdata):
    model = read_model(testdata / 'nonmem' / 'minimal.mod')
    assert model.parameters['THETA(1)'].init == 0.1
    model2 = read_model(str(testdata / 'nonmem' / 'minimal.mod'))
    assert model2.parameters['THETA(1)'].init == 0.1


def test_read_model_from_string(testdata):
    model = read_model_from_string(
        """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
Y = THETA(1) + ETA(1) + ERR(1)

$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
    )
    assert model.parameters['THETA(1)'].init == 0.1


def test_write_model(testdata):
    model = read_model(testdata / 'nonmem' / 'minimal.mod')
    with Patcher(additional_skip_names=['pkgutil']):
        write_model(model, 'run1.mod')
        assert Path('run1.mod').is_file()


def test_fix_parameters(testdata):
    model = Model(testdata / 'nonmem' / 'minimal.mod')
    assert not model.parameters['THETA(1)'].fix
    fix_parameters(model, ['THETA(1)'])
    assert model.parameters['THETA(1)'].fix

    model = Model(testdata / 'nonmem' / 'minimal.mod')
    assert not model.parameters['THETA(1)'].fix
    fix_parameters(model, 'THETA(1)')
    assert model.parameters['THETA(1)'].fix


def test_unfix_parameters(testdata):
    model = Model(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters(model, ['THETA(1)'])
    assert model.parameters['THETA(1)'].fix
    unfix_parameters(model, ['THETA(1)'])
    assert not model.parameters['THETA(1)'].fix

    model = Model(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters(model, 'THETA(1)')
    assert model.parameters['THETA(1)'].fix
    unfix_parameters(model, 'THETA(1)')
    assert not model.parameters['THETA(1)'].fix


def test_fix_parameters_to(testdata):
    model = Model(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters_to(model, 'THETA(1)', 0)
    assert model.parameters['THETA(1)'].fix
    assert model.parameters['THETA(1)'].init == 0

    model = Model(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters_to(model, ['THETA(1)', 'OMEGA(1,1)'], 0)
    assert model.parameters['THETA(1)'].fix
    assert model.parameters['THETA(1)'].init == 0
    assert model.parameters['THETA(1)'].fix
    assert model.parameters['OMEGA(1,1)'].init == 0

    model = Model(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters_to(model, ['THETA(1)', 'OMEGA(1,1)'], [0, 1])
    assert model.parameters['THETA(1)'].init == 0
    assert model.parameters['OMEGA(1,1)'].init == 1

    model = Model(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters_to(model, None, 0)
    assert all(p.fix for p in model.parameters)
    assert all(p.init == 0 for p in model.parameters)

    model = Model(testdata / 'nonmem' / 'minimal.mod')
    with pytest.raises(ValueError, match='Incorrect number of values'):
        fix_parameters_to(model, ['THETA(1)', 'OMEGA(1,1)'], [0, 0, 0])

    model = Model(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters_to(model, None, float(0))
    assert model.parameters['THETA(1)'].fix
    assert model.parameters['THETA(1)'].init == 0


def test_unfix_parameters_to(testdata):
    model = Model(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters(model, ['THETA(1)'])
    assert model.parameters['THETA(1)'].fix
    unfix_parameters_to(model, 'THETA(1)', 0)
    assert not model.parameters['THETA(1)'].fix
    assert model.parameters['THETA(1)'].init == 0

    model = Model(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters(model, ['THETA(1)', 'OMEGA(1,1)'])
    assert model.parameters['THETA(1)'].fix
    assert model.parameters['OMEGA(1,1)'].fix
    unfix_parameters_to(model, ['THETA(1)', 'OMEGA(1,1)'], 0)
    assert not model.parameters['THETA(1)'].fix
    assert not model.parameters['OMEGA(1,1)'].fix
    assert model.parameters['THETA(1)'].init == 0
    assert model.parameters['OMEGA(1,1)'].init == 0

    unfix_parameters_to(model, None, values=[1, 2, 3])


def test_generate_model_code(testdata):
    model = Model(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters(model, ['THETA(1)'])
    assert generate_model_code(model).split('\n')[7] == '$THETA 0.1 FIX'


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
            '$ESTIMATION METHOD=ZERO INTER MAXEVAL=9990 SADDLE_RESET=1',
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
    model = Model(testdata / 'nonmem' / 'minimal.mod')
    set_estimation_step(model, method, **kwargs)
    assert generate_model_code(model).split('\n')[-2] == code_ref


def test_add_estimation_step(testdata):
    model = Model(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    add_estimation_step(model, 'fo')
    assert len(model.estimation_steps) == 2
    assert generate_model_code(model).split('\n')[-2] == '$ESTIMATION METHOD=ZERO'

    model = Model(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    add_estimation_step(model, 'fo', evaluation=True)
    assert len(model.estimation_steps) == 2
    assert generate_model_code(model).split('\n')[-2] == '$ESTIMATION METHOD=ZERO MAXEVAL=0'


def test_add_estimation_step_non_int(testdata):
    model = Model(testdata / 'nonmem' / 'minimal.mod')
    add_estimation_step(model, 'fo', idx=1.0)
    with pytest.raises(TypeError, match='Index must be integer'):
        add_estimation_step(model, 'fo', idx=1.5)


def test_remove_estimation_step(testdata):
    model = Model(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    remove_estimation_step(model, 0)
    assert not model.estimation_steps
    assert generate_model_code(model).split('\n')[-2] == '$SIGMA 1'


def test_append_estimation_step_options(testdata):
    model = Model(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    append_estimation_step_options(model, {'SADDLE_RESET': 1}, 0)
    assert (
        model.model_code.split('\n')[-2]
        == '$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC SADDLE_RESET=1'
    )


def test_load_example_model():
    model = load_example_model("pheno")
    assert len(model.parameters) == 6
    assert len(model.modelfit_results.parameter_estimates) == 6

    with pytest.raises(ValueError):
        load_example_model("grekztalb23=")


def test_get_model_covariates(testdata):
    model = load_example_model("pheno")
    assert set(get_model_covariates(model)) == {sympy.Symbol('WGT'), sympy.Symbol('APGR')}
    minimal = Model(testdata / 'nonmem' / 'minimal.mod')
    assert set(get_model_covariates(minimal)) == set()


def test_set_name():
    model = load_example_model("pheno")
    set_name(model, "run1")
    assert model.name == "run1"


def test_copy_model():
    model = load_example_model("pheno")
    run1 = copy_model(model)
    assert id(model) != id(run1)
    assert id(model.parameters) != id(run1.parameters)
    run2 = copy_model(run1, "run2")
    assert run2.name == "run2"


def test_convert_model():
    model = load_example_model("pheno")
    run1 = convert_model(model, "nlmixr")
    assert model.name == run1.name
