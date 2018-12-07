import pytest
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy as np

import pharmpy.data as data


def test_resampler():
    np.random.seed(28)
    df = pd.DataFrame({'ID': [1, 1, 2, 2, 4, 4], 'DV': [5, 6, 3, 4, 0, 9], 'STRAT': [1, 1, 2, 2, 2, 2]})
    resampler = data.Resampler(df, 'ID')
    (new_df, ids) = next(resampler)
    assert list(ids) == [1, 4, 2]
    assert list(new_df['ID']) == [1, 1, 2, 2, 3, 3]
    assert list(new_df['DV']) == [5, 6, 0, 9, 3, 4]

    with pytest.raises(ValueError) as e:
        resampler = data.Resampler(df, 'ID', replace=False, sample_size=4)
        next(resample)

    resampler = data.Resampler(df, 'ID', replace=False, sample_size=3)
    next(resampler)

    resampler = data.Resampler(df, 'ID', stratify='STRAT')
    (new_df, ids) = next(resampler)
    assert list(ids) == [1, 4, 2]
    assert list(new_df['ID']) == [1, 1, 2, 2, 3, 3]
    assert list(new_df['DV']) == [5, 6, 0, 9, 3, 4]
