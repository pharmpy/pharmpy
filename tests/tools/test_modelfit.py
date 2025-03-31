import numpy as np
import pytest

from pharmpy.basic import Expr
from pharmpy.tools.modelfit.evaluation import (
    evaluate_model,
    get_functions_to_solve_for,
    get_variables_before_odes,
)
from pharmpy.tools.modelfit.ucp import (
    build_initial_values_matrix,
    build_parameter_coordinates,
    calculate_matrix_gradient_scale,
    calculate_theta_gradient_scale,
    descale_matrix,
    descale_thetas,
    scale_matrix,
    scale_thetas,
    split_ucps,
    unpack_ucp_matrix,
)


def test_scale_matrix():
    omega = np.array(
        [
            [0.0750, 0.0467, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0467, 0.0564, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.82, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0147, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0147, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.506, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.506],
        ]
    )
    S = scale_matrix(omega)
    est1 = np.array(
        [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
        ]
    )
    desc = descale_matrix(est1, S)
    assert np.allclose(desc, omega)


def test_build_initial_values_matrix(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox1.mod')
    omega = build_initial_values_matrix(model.random_variables.etas, model.parameters)
    correct = np.array([[0.075, 0.0467, 0.0], [0.0467, 0.0564, 0.0], [0.0, 0.0, 2.82]])
    assert np.allclose(omega, correct)


def test_build_parameter_coordinates(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox1.mod')
    omega = build_initial_values_matrix(model.random_variables.etas, model.parameters)
    coords = build_parameter_coordinates(omega)
    assert coords == [(0, 0), (1, 0), (1, 1), (2, 2)]
    sigma = build_initial_values_matrix(model.random_variables.epsilons, model.parameters)
    coords = build_parameter_coordinates(sigma)
    assert coords == [(0, 0)]


def test_unpack_ucp_matrix():
    coords = [(0, 0), (1, 0), (1, 1), (2, 2)]
    A = unpack_ucp_matrix([1.0, 2.0, 3.0, 4.0], coords)
    correct = np.array([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    assert np.array_equal(A, correct)


def test_split_ucps(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox1.mod')
    omega = build_initial_values_matrix(model.random_variables.etas, model.parameters)
    omega_coords = build_parameter_coordinates(omega)
    sigma = build_initial_values_matrix(model.random_variables.epsilons, model.parameters)
    sigma_coords = build_parameter_coordinates(sigma)
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    thetas, omegas, sigmas = split_ucps(x, omega_coords, sigma_coords)
    assert np.array_equal(thetas, [1.0, 2.0, 3.0])
    assert np.array_equal(omegas, [[4.0, 0.0, 0.0], [5.0, 6.0, 0.0], [0.0, 0.0, 7.0]])
    assert np.array_equal(sigmas, [[8.0]])


def test_scale_thetas(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    parameters = model.parameters[(0, 1, 2)]
    scale = scale_thetas(parameters)
    x = descale_thetas([0.1, 0.1, 0.1], scale)
    assert np.allclose(x, np.array([0.00469307, 1.00916, 0.1]))

    x = descale_thetas([1.0053e-01, 7.5015e-02, 1.5264e-01], scale)
    assert np.allclose(x, np.array([4.6955e-03, 9.8426e-01, 1.5892e-01]), atol=1e-5)


def test_calculate_theta_gradient_scale(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    theta_ucp = np.array([0.1])
    scale = scale_thetas(model.parameters[(0,)])
    grad = calculate_theta_gradient_scale(theta_ucp, scale)
    assert list(grad) == [1.0]


def test_calculate_matrix_gradient_scale():
    omega_ucp = np.array([[0.1]])
    scale = scale_matrix(omega_ucp)
    grad = calculate_matrix_gradient_scale(omega_ucp, scale, build_parameter_coordinates(omega_ucp))
    assert np.allclose(grad, np.array([0.2]))


def test_get_variables_before_odes(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    vars = get_variables_before_odes(model)
    assert len(vars) == 3
    assert vars[Expr.symbol('CL')] == Expr("0.00469307*WGT*exp(ETA_CL)")


def test_get_functions_to_solve_for(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    vars = get_functions_to_solve_for(model)
    assert vars == {Expr.function("A_CENTRAL", "t")}


def test_evaluate_model(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    res = evaluate_model(model)
    print(res)
    assert res.loc[0, 'Y'] == pytest.approx(17.695056, abs=1e-5)
    assert res.loc[743, 'Y'] == pytest.approx(34.411508, abs=1e-5)
