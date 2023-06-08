from typing import Iterable

import sympy

from pharmpy.model import Assignment
from pharmpy.modeling import (
    add_peripheral_compartment,
    create_joint_distribution,
    get_initial_conditions,
    get_zero_order_inputs,
    has_linear_odes,
    has_linear_odes_with_real_eigenvalues,
    has_odes,
    remove_iiv,
    set_initial_condition,
    set_michaelis_menten_elimination,
    set_ode_solver,
    set_transit_compartments,
    set_zero_order_elimination,
    set_zero_order_input,
)
from pharmpy.modeling.odes import find_clearance_parameters, find_volume_parameters
from pharmpy.tools import read_modelfit_results


def test_nested_update_source(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    res = read_modelfit_results(pheno_path)

    model = create_joint_distribution(model, individual_estimates=res.individual_estimates)
    model = model.update_source()

    assert 'IIV_CL_IIV_V' in model.model_code

    model = load_model_for_test(pheno_path)

    model = remove_iiv(model, 'CL')

    model = model.update_source()

    assert '0.031128' in model.model_code
    assert '0.0309626' not in model.model_code

    model = load_model_for_test(pheno_path)

    model = remove_iiv(model, 'V')

    model = model.update_source()

    assert '0.0309626' in model.model_code
    assert '0.031128' not in model.model_code


def test_set_ode_solver(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    assert model.estimation_steps[0].solver is None
    assert 'ADVAN1' in model.model_code
    assert '$MODEL' not in model.model_code

    model = load_model_for_test(pheno_path)
    model = set_michaelis_menten_elimination(model)
    model = set_ode_solver(model, 'LSODA')
    assert model.estimation_steps[0].solver == 'LSODA'
    assert 'ADVAN13' in model.model_code
    assert '$MODEL' in model.model_code

    model = load_model_for_test(pheno_path)
    model = set_zero_order_elimination(model)
    assert 'ADVAN13' in model.model_code
    assert '$MODEL' in model.model_code
    model = set_ode_solver(model, 'LSODA')
    model = set_michaelis_menten_elimination(model)
    assert model.estimation_steps[0].solver == 'LSODA'
    assert 'ADVAN13' in model.model_code
    assert '$MODEL' in model.model_code
    model = set_ode_solver(model, 'DVERK')
    assert model.estimation_steps[0].solver == 'DVERK'
    assert 'ADVAN6' in model.model_code
    assert '$MODEL' in model.model_code


def _symbols(names: Iterable[str]):
    return list(map(sympy.Symbol, names))


def test_find_clearance_parameters(pheno):
    cl_origin = find_clearance_parameters(pheno)
    assert cl_origin == _symbols(['CL'])

    model = add_peripheral_compartment(pheno)
    cl_p1 = find_clearance_parameters(model)
    assert cl_p1 == _symbols(['CL', 'QP1'])

    model = add_peripheral_compartment(model)
    cl_p2 = find_clearance_parameters(model)
    assert cl_p2 == _symbols(['CL', 'QP1', 'QP2'])


def test_find_clearance_parameters_github_issues_1053_and_1062(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    model = set_michaelis_menten_elimination(model)
    assert find_clearance_parameters(model) == _symbols(['CLMM'])


def test_find_clearance_parameters_github_issues_1044_and_1053(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    model = set_transit_compartments(model, 10)
    assert find_clearance_parameters(model) == _symbols(['CL'])


def test_find_clearance_parameters_github_issues_1053_and_1062_bis(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    model = add_peripheral_compartment(model)
    model = add_peripheral_compartment(model)
    model = set_michaelis_menten_elimination(model)
    assert find_clearance_parameters(model) == _symbols(['CLMM', 'QP1', 'QP2'])


def test_find_volume_parameters(pheno):
    v_origin = find_volume_parameters(pheno)
    assert v_origin == _symbols(['V'])

    model = add_peripheral_compartment(pheno)
    v_p1 = find_volume_parameters(model)
    assert v_p1 == _symbols(['V1', 'VP1'])

    model = add_peripheral_compartment(model)
    v_p2 = find_volume_parameters(model)
    assert v_p2 == _symbols(['V1', 'VP1', 'VP2'])


def test_find_volume_parameters_github_issues_1053_and_1062(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    model = set_michaelis_menten_elimination(model)
    assert find_volume_parameters(model) == _symbols(['V'])


def test_find_volume_parameters_github_issues_1044_and_1053(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    model = set_transit_compartments(model, 10)
    assert find_volume_parameters(model) == _symbols(['V'])


def test_find_volume_parameters_github_issues_1053_and_1062_bis(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    model = add_peripheral_compartment(model)
    model = add_peripheral_compartment(model)
    model = set_michaelis_menten_elimination(model)
    assert find_volume_parameters(model) == _symbols(['V1', 'VP1', 'VP2'])


def test_has_odes(load_example_model_for_test, datadir, load_model_for_test):
    model = load_example_model_for_test('pheno')
    assert has_odes(model)
    path = datadir / 'minimal.mod'
    model = load_model_for_test(path)
    assert not has_odes(model)


def test_has_linear_odes(load_example_model_for_test, datadir, load_model_for_test):
    model = load_example_model_for_test('pheno')
    assert has_linear_odes(model)
    model = set_michaelis_menten_elimination(model)
    assert not has_linear_odes(model)
    path = datadir / 'minimal.mod'
    model = load_model_for_test(path)
    assert not has_linear_odes(model)


def test_has_linear_odes_with_real_eigenvalues(
    load_example_model_for_test, datadir, load_model_for_test
):
    model = load_example_model_for_test('pheno')
    assert has_linear_odes_with_real_eigenvalues(model)
    model = set_michaelis_menten_elimination(model)
    assert not has_linear_odes_with_real_eigenvalues(model)
    path = datadir / 'minimal.mod'
    model = load_model_for_test(path)
    assert not has_linear_odes_with_real_eigenvalues(model)


def test_get_initial_conditions(load_example_model_for_test, load_model_for_test, datadir):
    model = load_example_model_for_test('pheno')
    assert get_initial_conditions(model) == {sympy.Function('A_CENTRAL')(0): sympy.Integer(0)}
    ic = Assignment(sympy.Function('A_CENTRAL')(0), sympy.Integer(23))
    statements = (
        model.statements.before_odes
        + ic
        + model.statements.ode_system
        + model.statements.after_odes
    )
    mod2 = model.replace(statements=statements)
    assert get_initial_conditions(mod2) == {sympy.Function('A_CENTRAL')(0): sympy.Integer(23)}
    path = datadir / 'minimal.mod'
    model = load_model_for_test(path)
    assert get_initial_conditions(model) == {}


def test_set_intial_conditions(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    model = set_initial_condition(model, "CENTRAL", 10)
    assert len(model.statements) == 16
    ic = Assignment(sympy.Function('A_CENTRAL')(0), sympy.Integer(10))
    assert model.statements.before_odes[-1] == ic
    assert get_initial_conditions(model) == {sympy.Function('A_CENTRAL')(0): sympy.Integer(10)}
    model = set_initial_condition(model, "CENTRAL", 23)
    assert len(model.statements) == 16
    ic = Assignment(sympy.Function('A_CENTRAL')(0), sympy.Integer(23))
    assert model.statements.before_odes[-1] == ic
    model = set_initial_condition(model, "CENTRAL", 0)
    assert len(model.statements) == 15


def test_get_zero_order_inputs(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    zo = get_zero_order_inputs(model)
    assert zo == sympy.Matrix([[0]])


def test_set_zero_order_input(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    model = set_zero_order_input(model, "CENTRAL", 10)
    zo = get_zero_order_inputs(model)
    assert zo == sympy.Matrix([[10]])
