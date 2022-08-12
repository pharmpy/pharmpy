import sympy

from pharmpy.modeling import (
    add_population_parameter,
    fix_or_unfix_parameters,
    fix_parameters,
    fix_parameters_to,
    get_omegas,
    get_sigmas,
    get_thetas,
    set_lower_bounds,
    set_upper_bounds,
    unconstrain_parameters,
    unfix_parameters,
    unfix_parameters_to,
)


def test_get_thetas(pheno):
    thetas = get_thetas(pheno)
    assert len(thetas) == 3
    assert thetas['THETA(1)'].init == 0.00469307


def test_get_omegas(pheno):
    omegas = get_omegas(pheno)
    assert len(omegas) == 2
    assert omegas['OMEGA(1,1)'].init == 0.0309626


def test_get_sigmas(pheno):
    sigmas = get_sigmas(pheno)
    assert len(sigmas) == 1
    assert sigmas['SIGMA(1,1)'].init == 0.013241


def test_fix_parameters(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert not model.parameters['THETA(1)'].fix
    fix_parameters(model, ['THETA(1)'])
    assert model.parameters['THETA(1)'].fix

    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert not model.parameters['THETA(1)'].fix
    fix_parameters(model, 'THETA(1)')
    assert model.parameters['THETA(1)'].fix


def test_unfix_parameters(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters(model, ['THETA(1)'])
    assert model.parameters['THETA(1)'].fix
    unfix_parameters(model, ['THETA(1)'])
    assert not model.parameters['THETA(1)'].fix

    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters(model, 'THETA(1)')
    assert model.parameters['THETA(1)'].fix
    unfix_parameters(model, 'THETA(1)')
    assert not model.parameters['THETA(1)'].fix


def test_fix_or_unfix_parameters(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    fix_or_unfix_parameters(model, {'THETA(1)': True})
    assert model.parameters['THETA(1)'].fix
    fix_or_unfix_parameters(model, {'THETA(1)': False})
    assert not model.parameters['THETA(1)'].fix


def test_unconstrain_parameters(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    fix_or_unfix_parameters(model, {'THETA(1)': True})
    unconstrain_parameters(model, ['THETA(1)'])
    assert not model.parameters['THETA(1)'].fix
    fix_or_unfix_parameters(model, {'THETA(1)': True})
    unconstrain_parameters(model, 'THETA(1)')
    assert not model.parameters['THETA(1)'].fix


def test_fix_parameters_to(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters_to(model, {'THETA(1)': 0})
    assert model.parameters['THETA(1)'].fix
    assert model.parameters['THETA(1)'].init == 0

    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters_to(model, {'THETA(1)': 0, 'OMEGA(1,1)': 0})
    assert model.parameters['THETA(1)'].fix
    assert model.parameters['THETA(1)'].init == 0
    assert model.parameters['THETA(1)'].fix
    assert model.parameters['OMEGA(1,1)'].init == 0

    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters_to(model, {'THETA(1)': 0, 'OMEGA(1,1)': 1})
    assert model.parameters['THETA(1)'].init == 0
    assert model.parameters['OMEGA(1,1)'].init == 1


def test_unfix_parameters_to(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters(model, ['THETA(1)'])
    assert model.parameters['THETA(1)'].fix
    unfix_parameters_to(model, {'THETA(1)': 0})
    assert not model.parameters['THETA(1)'].fix
    assert model.parameters['THETA(1)'].init == 0

    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters(model, ['THETA(1)', 'OMEGA(1,1)'])
    assert model.parameters['THETA(1)'].fix
    assert model.parameters['OMEGA(1,1)'].fix
    unfix_parameters_to(model, {'THETA(1)': 0, 'OMEGA(1,1)': 0})
    assert not model.parameters['THETA(1)'].fix
    assert not model.parameters['OMEGA(1,1)'].fix
    assert model.parameters['THETA(1)'].init == 0
    assert model.parameters['OMEGA(1,1)'].init == 0


def test_set_bounds(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    set_upper_bounds(model, {'THETA(1)': 100})
    assert model.parameters['THETA(1)'].upper == 100
    assert model.parameters['OMEGA(1,1)'].upper == sympy.oo
    set_lower_bounds(model, {'THETA(1)': -100})
    assert model.parameters['THETA(1)'].lower == -100
    assert model.parameters['OMEGA(1,1)'].lower == 0


def test_add_population_parameter(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    add_population_parameter(model, 'NEWPARAM', 23)
    assert len(model.parameters) == 4
    assert model.parameters['NEWPARAM'].init == 23
