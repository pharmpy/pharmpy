import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.df import pandas_from_dict, pandas_to_dict


@pytest.mark.parametrize(
    'obj',
    [
        pd.DataFrame({'DV': [1.0, 2.0]}),
        pd.Series(
            [1, 2, 3, 4, 5],
            name="MYSER",
            index=pd.Index([4, 8, 16, 32, 64], dtype='int32', name="MYIND"),
        ),
        pd.DataFrame(
            {"metric": [10.5, 20.3]},
            index=pd.MultiIndex.from_arrays(
                [
                    np.array([1.1, 2.2], dtype="float64"),
                    np.array([100, 200], dtype="int32"),
                ],
                names=["float_level", "int_level"],
            ),
        ),
        pd.Series(
            [100, 200],
            index=pd.MultiIndex.from_arrays(
                [
                    np.array([0.5, 1.5], dtype="float64"),
                    np.array([10, 20], dtype="int64"),
                ],
                names=["float_lvl", "int_lvl"],
            ),
            name="my_series",
        ),
    ],
)
def test_round_trip(obj):
    d = pandas_to_dict(obj)
    out_obj = pandas_from_dict(d)
    if isinstance(obj, pd.DataFrame):
        func = assert_frame_equal
    else:
        func = assert_series_equal
    func(out_obj, obj, check_exact=True)
