import pytest
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy as np

import pharmpy.data as data


def test_resample():
    np.random.seed(28)
    df = pd.DataFrame({'ID': [1, 1, 2, 2, 4, 4], 'DV': [5, 6, 3, 4, 0, 9], 'STRAT': [1, 1, 2, 2, 2, 2]})
    (new_df, ids) = data.resample(df, 'ID')
    assert list(ids) == [1, 4, 2]
    assert list(new_df['ID']) == [1, 1, 2, 2, 3, 3]
    assert list(new_df['DV']) == [5, 6, 0, 9, 3, 4]

    with pytest.raises(ValueError) as e:
        data.resample(df, 'ID', replace=True, sample_size=4)

    data.resample(df, 'ID', replace=True, sample_size=3)    # Should not raise

    (new_df, ids) = data.resample(df, 'ID', stratify='STRAT')
    assert list(ids) == [1, 4, 2]
    assert list(new_df['ID']) == [1, 1, 2, 2, 3, 3]
    assert list(new_df['DV']) == [5, 6, 0, 9, 3, 4]
