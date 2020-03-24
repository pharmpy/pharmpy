import numpy as np
import pandas as pd
import pytest
import sympy
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal

import pharmpy.math


def test_triangular_root():
    assert pharmpy.math.triangular_root(1) == 1
    assert pharmpy.math.triangular_root(3) == 2
    assert pharmpy.math.triangular_root(6) == 3
    assert pharmpy.math.triangular_root(10) == 4


def test_flattened_to_symmetric():
    assert_array_equal(pharmpy.math.flattened_to_symmetric([1.0]), np.array([[1.0]]))
    assert_array_equal(pharmpy.math.flattened_to_symmetric([1.0, 1.5, 2.0]),
                       np.array([[1.0, 1.5], [1.5, 2.0]]))
    A = pharmpy.math.flattened_to_symmetric([1.0, 1.5, 2.0, -1.0, 3.0, 5.5])
    assert_array_equal(A, np.array([[1.0, 1.5, -1.0], [1.5, 2.0, 3.0], [-1.0, 3.0, 5.5]]))


def test_round_and_keep_sum():
    ser = pd.Series([1.052632, 0.701754, 0.701754, 1.052632, 2.456140,
                     2.807018, 4.210526, 4.210526, 3.157895, 0.350877])
    correct_results = pd.Series([1, 1, 1, 1, 2, 3, 4, 4, 3, 0])
    rounded = pharmpy.math.round_and_keep_sum(ser, 20)
    assert_series_equal(rounded, correct_results)


def test_se_delta_method():
    vals = {'OMEGA(1,1)': 3.75637E-02, 'OMEGA(2,1)': 1.93936E-02, 'OMEGA(2,2)': 2.19133E-02}
    om11 = sympy.Symbol('OMEGA(1,1)')
    om21 = sympy.Symbol('OMEGA(2,1)')
    om22 = sympy.Symbol('OMEGA(2,2)')
    expr = om21 / (sympy.sqrt(om11) * sympy.sqrt(om22))
    names = ['OMEGA(1,1)', 'OMEGA(2,1)', 'OMEGA(2,2)']
    cov = pd.DataFrame([[4.17213E-04, 1.85060E-04, -3.51477E-05],
                        [1.85060E-04, 1.10836E-04, 3.61663E-06],
                        [-3.51477E-05, 3.61663E-06, 4.44030E-05]], columns=names, index=names)
    se = pharmpy.math.se_delta_method(expr, vals, cov)
    assert pytest.approx(0.2219739865800438, 1e-15) == se


def test_is_posdef():
    A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    assert pharmpy.math.is_posdef(A)
    B = np.array([[1, 2], [2, 1]])
    assert not pharmpy.math.is_posdef(B)


def test_nearest_posdef():
    for i in range(5):
        for j in range(2, 20):
            A = np.random.randn(j, j)
            B = pharmpy.math.nearest_posdef(A)
            assert pharmpy.math.is_posdef(B)


def test_sample_truncated_joint_normal():
    samples = pharmpy.math.sample_truncated_joint_normal(
        np.array([0, 0]), np.array([[2, 0.1], [0.1, 1]]), np.array([-1, -2]), np.array([1, 2]), 10)
    assert (samples[:, 0] > -1).all()
    assert (samples[:, 0] < 1).all()
    assert (samples[:, 1] > -2).all()
    assert (samples[:, 1] < 2).all()
