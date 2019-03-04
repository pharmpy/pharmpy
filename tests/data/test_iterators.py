import pytest
import unittest.mock as mock
import pandas as pd
import numpy as np

import pharmpy.data.iterators as iters


@pytest.fixture
def df():
    return pd.DataFrame({'ID': [1, 1, 2, 2, 4, 4], 'DV': [5, 6, 3, 4, 0, 9], 'STRAT': [1, 1, 2, 2, 2, 2]})


class MyDefaultIterator(iters.DatasetIterator):
    def __next__(self):
        try:
            self.i
        except:
            self.i = 0
        if self.i == 2:
            raise StopIteration
        else:
            self.i += 1
            return mock.Mock()


class MyTupleIterator(iters.DatasetIterator):
    def __next__(self):
        try:
            self.i
        except:
            self.i = 0
        if self.i == 2:
            raise StopIteration
        else:
            self.i += 1
            return mock.Mock(), "DUMMY"

    def data_files(self, path, filename="my_{}.csv"):
        return super().data_files(path, filename)


def test_base_class(df):
    it = MyDefaultIterator()
    files = it.data_files(path='/anywhere')
    assert [str(x) for x in files] == ['/anywhere/dataset_1.dta', '/anywhere/dataset_2.dta']

    it = MyTupleIterator()
    files = it.data_files(path='/nowhere')
    assert [str(x) for x in files] == ['/nowhere/my_1.csv', '/nowhere/my_2.csv']


def test_omit(df):
    omitter = iters.Omit(df, 'ID')
    (new_df, group) = next(omitter)
    assert group == 1
    assert list(new_df['ID']) == [2, 2, 4, 4]
    assert list(new_df['DV']) == [3, 4, 0, 9]
    (new_df, group) = next(omitter)
    assert group == 2
    assert list(new_df['ID']) == [1, 1, 4, 4]
    assert list(new_df['DV']) == [5, 6, 0, 9]
    (new_df, group) = next(omitter)
    assert group == 4
    assert list(new_df['ID']) == [1, 1, 2, 2]
    assert list(new_df['DV']) == [5, 6, 3, 4]
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
    with pytest.raises(StopIteration):      # Test the default one iteration
        next(resampler)


def test_resampler_too_big_sample_size(df):
    with pytest.raises(ValueError) as e:
        resampler = iters.Resample(df, 'ID', replace=False, sample_size=4)
        next(resample)


def test_resampler_noreplace(df):
    resampler = iters.Resample(df, 'ID', replace=False, sample_size=3)
    next(resampler)

    resampler = iters.Resample(df, 'ID', stratify='STRAT')
    (new_df, ids) = next(resampler)
    assert ids == [1, 4, 2]
    assert list(new_df['ID']) == [1, 1, 2, 2, 3, 3]
    assert list(new_df['DV']) == [5, 6, 0, 9, 3, 4]

    resampler = iters.Resample(df, 'ID', replace=False, sample_size=2)
    (new_df, ids) = next(resampler)
    assert list(ids) == [4, 2]
    assert list(new_df['ID']) == [1, 1, 2, 2]


def test_stratification(df):
    resampler = iters.Resample(df, 'ID', resamples=1, stratify='STRAT', sample_size={1: 2, 2: 3}, replace=True)
    (new_df, ids) = next(resampler)
    assert ids == [1, 1, 4, 2, 2]
    assert list(new_df['ID']) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert list(new_df['DV']) == [5, 6, 5, 6, 0, 9, 3, 4, 3, 4]
    
    resampler = iters.Resample(df, 'ID', resamples=1, stratify='STRAT', sample_size={1: 2}, replace=True)
    (new_df, ids) = next(resampler)
    assert ids == [1, 1]

    resampler = iters.Resample(df, 'ID', resamples=3, stratify='STRAT', replace=True)
    (new_df, ids) = next(resampler)
    assert ids == [1, 4, 4]
    (new_df, ids) = next(resampler)
    assert ids == [1, 2, 2]
    (new_df, ids) = next(resampler)
    assert ids == [1, 4, 2]
