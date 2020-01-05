import numpy as np
import pandas as pd
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
