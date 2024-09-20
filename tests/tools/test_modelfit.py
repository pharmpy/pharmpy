import numpy as np

from pharmpy.tools.modelfit.ucp import (
    build_initial_values_matrix,
    build_parameter_coordinates,
    descale_matrix,
    scale_matrix,
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
    A = unpack_ucp_matrix([1.0, 2.0, 3.0, 4.0], 3, coords)
    correct = np.array([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    assert np.array_equal(A, correct)
