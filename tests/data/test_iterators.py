import numpy as np
import pandas.testing
import pytest

import pharmpy.data
import pharmpy.data.iterators as iters


@pytest.fixture
def df():
    return pharmpy.data.PharmDataFrame(
        {'ID': [1, 1, 2, 2, 4, 4], 'DV': [5, 6, 3, 4, 0, 9], 'STRAT': [1, 1, 2, 2, 2, 2]}
    )


def test_omit(df):
    omitter = iters.Omit(df, 'ID')
    (new_df, group) = next(omitter)
    assert group == 1
    assert list(new_df['ID']) == [2, 2, 4, 4]
    assert list(new_df['DV']) == [3, 4, 0, 9]
    assert new_df.name == 'omitted_1'
    assert not hasattr(df, 'name')  # Check that original did not get name
    (new_df, group) = next(omitter)
    assert group == 2
    assert list(new_df['ID']) == [1, 1, 4, 4]
    assert list(new_df['DV']) == [5, 6, 0, 9]
    assert new_df.name == 'omitted_2'
    (new_df, group) = next(omitter)
    assert group == 4
    assert list(new_df['ID']) == [1, 1, 2, 2]
    assert list(new_df['DV']) == [5, 6, 3, 4]
    assert new_df.name == 'omitted_3'
    with pytest.raises(StopIteration):
        next(omitter)


def test_resampler_default(df):
    np.random.seed(28)
    resampler = iters.Resample(df, 'ID')
    (new_df, ids) = next(resampler)
    assert ids == [1, 4, 2]
    assert list(new_df['ID']) == [1, 1, 2, 2, 3, 3]
    assert list(new_df['DV']) == [5, 6, 0, 9, 3, 4]
    assert list(new_df['STRAT']) == [1, 1, 2, 2, 2, 2]
    assert new_df.name == 'resample_1'
    with pytest.raises(StopIteration):  # Test the default one iteration
        next(resampler)


def test_resampler_too_big_sample_size(df):
    with pytest.raises(ValueError):
        iters.Resample(df, 'ID', replace=False, sample_size=4)


def test_resampler_noreplace(df):
    np.random.seed(28)
    resampler = iters.Resample(df, 'ID', replace=False, sample_size=3)
    next(resampler)

    resampler = iters.Resample(df, 'ID', stratify='STRAT')
    (new_df, ids) = next(resampler)
    assert ids == [1, 2, 4]
    assert list(new_df['ID']) == [1, 1, 2, 2, 3, 3]
    assert list(new_df['DV']) == [5, 6, 3, 4, 0, 9]

    resampler = iters.Resample(df, 'ID', replace=False, sample_size=2)
    (new_df, ids) = next(resampler)
    assert list(ids) == [2, 1]
    assert list(new_df['ID']) == [1, 1, 2, 2]


def test_stratification(df):
    np.random.seed(28)
    resampler = iters.Resample(
        df, 'ID', resamples=1, stratify='STRAT', sample_size={1: 2, 2: 3}, replace=True
    )
    (new_df, ids) = next(resampler)
    assert ids == [1, 1, 4, 4, 4]
    assert list(new_df['ID']) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert list(new_df['DV']) == [5, 6, 5, 6, 0, 9, 0, 9, 0, 9]

    resampler = iters.Resample(
        df, 'ID', resamples=1, stratify='STRAT', sample_size={1: 2}, replace=True
    )
    (new_df, ids) = next(resampler)
    assert ids == [1, 1]

    resampler = iters.Resample(df, 'ID', resamples=3, stratify='STRAT', replace=True)
    (new_df, ids) = next(resampler)
    assert ids == [1, 2, 2]
    (new_df, ids) = next(resampler)
    assert ids == [1, 2, 4]
    (new_df, ids) = next(resampler)
    assert ids == [1, 4, 2]


def test_resampler_anonymization(testdata):
    np.random.seed(28)
    df = pharmpy.data.read_csv(testdata / 'pheno_data.csv')
    resampler = iters.Resample(df, group='ID')
    (new_df, ids) = next(resampler)
    assert all(e in ids for e in range(1, 60))
    assert len(ids) == 59
    for new_id, old_id in enumerate(ids, start=1):
        df_newid = new_df.query(f'ID == {new_id}')
        df_oldid = df.query(f'ID == {old_id}')
        df_newid = df_newid.reset_index(drop=True)
        df_oldid.reset_index(inplace=True, drop=True)
        df_newid['ID'] = old_id
        pandas.testing.assert_frame_equal(df_newid, df_oldid)
