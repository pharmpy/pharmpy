import warnings

import numpy as np
import pandas as pd
import pytest
import sympy

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from numpy.testing import assert_allclose, assert_array_equal

from pandas.testing import assert_series_equal

from pharmpy.internals.math import (
    conditional_joint_normal,
    conditional_joint_normal_lambda,
    corr2cov,
    cov2corr,
    flattened_to_symmetric,
    is_posdef,
    is_positive_semidefinite,
    nearest_postive_semidefinite,
    round_and_keep_sum,
    round_to_n_sigdig,
    se_delta_method,
    triangular_root,
)


def test_triangular_root():
    assert triangular_root(1) == 1
    assert triangular_root(3) == 2
    assert triangular_root(6) == 3
    assert triangular_root(10) == 4


def test_flattened_to_symmetric():
    assert_array_equal(flattened_to_symmetric([1.0]), np.array([[1.0]]))
    assert_array_equal(flattened_to_symmetric([1.0, 1.5, 2.0]), np.array([[1.0, 1.5], [1.5, 2.0]]))
    A = flattened_to_symmetric([1.0, 1.5, 2.0, -1.0, 3.0, 5.5])
    assert_array_equal(A, np.array([[1.0, 1.5, -1.0], [1.5, 2.0, 3.0], [-1.0, 3.0, 5.5]]))


def test_round_to_n_sigdig():
    assert round_to_n_sigdig(12345, 3) == 12300
    assert round_to_n_sigdig(23.45, 1) == 20
    assert round_to_n_sigdig(-0.012645, 2) == -0.013
    assert round_to_n_sigdig(0, 0) == 0
    assert round_to_n_sigdig(0, 1) == 0
    assert round_to_n_sigdig(0, 2) == 0


def test_round_and_keep_sum():
    ser = pd.Series(
        [
            1.052632,
            0.701754,
            0.701754,
            1.052632,
            2.456140,
            2.807018,
            4.210526,
            4.210526,
            3.157895,
            0.350877,
        ]
    )
    correct_results = pd.Series([1, 1, 1, 1, 2, 3, 4, 4, 3, 0])
    rounded = round_and_keep_sum(ser, 20)
    assert_series_equal(rounded, correct_results)

    ser = pd.Series([1.0])
    rounded = round_and_keep_sum(ser, 1.0)
    assert_series_equal(rounded, pd.Series([1]))

    rounded = round_and_keep_sum(pd.Series([], dtype=np.float64), 1.0)
    assert_series_equal(rounded, pd.Series([], dtype=np.int64))


def test_corr2cov():
    corr = np.array([[1.00, 0.25, 0.90], [0.25, 1.00, 0.50], [0.90, 0.50, 1.00]])
    sd = np.array([1, 4, 9])
    cov = corr2cov(corr, sd)
    assert_array_equal(cov, np.array([[1, 1, 8.1], [1, 16, 18], [8.1, 18, 81]]))


def test_cov2corr():
    cov = np.array([[1.0, 1.0, 8.1], [1.0, 16.0, 18.0], [8.1, 18.0, 81.0]])
    corr = cov2corr(cov)
    assert_allclose(corr, np.array([[1.0, 0.25, 0.9], [0.25, 1, 0.5], [0.9, 0.5, 1]]))


def test_se_delta_method():
    vals = {'OMEGA(1,1)': 3.75637e-02, 'OMEGA(2,1)': 1.93936e-02, 'OMEGA(2,2)': 2.19133e-02}
    om11 = sympy.Symbol('OMEGA(1,1)')
    om21 = sympy.Symbol('OMEGA(2,1)')
    om22 = sympy.Symbol('OMEGA(2,2)')
    expr = om21 / (sympy.sqrt(om11) * sympy.sqrt(om22))
    names = ['OMEGA(1,1)', 'OMEGA(2,1)', 'OMEGA(2,2)']
    cov = pd.DataFrame(
        [
            [4.17213e-04, 1.85060e-04, -3.51477e-05],
            [1.85060e-04, 1.10836e-04, 3.61663e-06],
            [-3.51477e-05, 3.61663e-06, 4.44030e-05],
        ],
        columns=names,
        index=names,
    )
    se = se_delta_method(expr, vals, cov)
    assert pytest.approx(0.2219739865800438, 1e-15) == se


def test_is_posdef():
    A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    assert is_posdef(A)
    B = np.array([[1, 2], [2, 1]])
    assert not is_posdef(B)


def test_nearest_posdef():
    for _ in range(5):
        for j in range(2, 20):
            A = np.random.randn(j, j)
            B = nearest_postive_semidefinite(A)
            assert is_positive_semidefinite(B)


def test_conditional_joint_normal():
    sigma = [
        [0.0419613930249351, 0.0194493895550238, -0.00815616219453746, 0.0943578658777171],
        [0.0194493895550238, 0.0296333601234358, 0.107516199715367, 0.0353748349332184],
        [-0.00815616219453746, 0.107516199715367, 0.883267716518442, 0.101648158864576],
        [0.0943578658777171, 0.0353748349332184, 0.101648158864576, 0.887220758887173],
    ]
    sigma = np.array(sigma)

    scaling = np.diag(np.array([1, 1, 2.2376, 0.70456]))
    scaled_sigma = scaling @ sigma @ scaling.T

    WGT_mean = 1.52542372881356
    WGT_5th = 0.7
    WGT_sigma = scaled_sigma[[0, 1, 3]][:, [0, 1, 3]]
    mu = [0, 0, WGT_mean]

    mu_bar, _ = conditional_joint_normal(mu, WGT_sigma, np.array([WGT_5th]))

    np.testing.assert_array_almost_equal(mu_bar, np.array([-0.12459637, -0.04671127]))


def test_conditional_joint_normal_lambda():
    sigma = [
        [0.0419613930249351, 0.0194493895550238, -0.00815616219453746, 0.0943578658777171],
        [0.0194493895550238, 0.0296333601234358, 0.107516199715367, 0.0353748349332184],
        [-0.00815616219453746, 0.107516199715367, 0.883267716518442, 0.101648158864576],
        [0.0943578658777171, 0.0353748349332184, 0.101648158864576, 0.887220758887173],
    ]
    sigma = np.array(sigma)

    scaling = np.diag(np.array([1, 1, 2.2376, 0.70456]))
    scaled_sigma = scaling @ sigma @ scaling.T

    WGT_mean = 1.52542372881356
    WGT_5th = 0.7
    WGT_sigma = scaled_sigma[[0, 1, 3]][:, [0, 1, 3]]
    mu = [0, 0, WGT_mean]

    mu_1, sigma_1 = conditional_joint_normal(mu, WGT_sigma, np.array([WGT_5th]))
    mu_2, sigma_2 = conditional_joint_normal_lambda(mu, WGT_sigma, len(mu) - 1)(np.array([WGT_5th]))

    assert (mu_1 == mu_2).all()
    assert (sigma_1 == sigma_2).all()
