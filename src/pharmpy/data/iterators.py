"""
data.iterators
==============

Iterators generating new datasets from a dataset. The dataset could either be stand alone
or connected to a model. If a model is used the same model will be updated with different
datasets for each iteration.

Currenly contains:

1. Omit - Can be used for cdd
2. Resample - Can be used by bootstrap
"""

import collections
import warnings

import numpy as np
import pandas as pd

import pharmpy.math


class DatasetIterator:
    """Base class for iterator classes that generate new datasets from an input dataset

    The __next__ function could return either a DataFrame or a tuple where the first
    element is the main DataFrame.
    """

    def __init__(self, iterations, name_pattern='dataset_{}'):
        """Initialization of the base class
        :param iterations: is the number of iterations
        :param name_pattern: Name pattern to use for generated datasets.
             A number starting from 1 will be put in the placeholder.
        """
        self._next = 1
        self._iterations = iterations
        self._name_pattern = name_pattern

    def _check_exhausted(self):
        """Check if the iterator is exhaused. Raise StopIteration in that case"""
        if self._next > self._iterations:
            raise StopIteration

    def _prepare_next(self, df):
        df.name = self._name_pattern.format(self._next)
        self._next += 1

    def _retrieve_dataset(self, df_or_model):
        """Extract dataset from model and remember the model.
        If input is dataset simply pass through and set model to None
        """
        try:
            dataset = df_or_model.dataset
            self._model = df_or_model
            return dataset
        except AttributeError:
            self._model = None
            return df_or_model

    def _combine_dataset(self, df):
        """If we are working with a model set the dataset and return model
        else simply pass the dataset through
        """
        try:
            model = self._model.copy()
        except AttributeError:
            return df
        else:
            model.dataset = df
            model.name = df.name
            return model

    def __iter__(self):
        return self


class Omit(DatasetIterator):
    """Iterate over omissions of a certain group in a dataset. One group is omitted at a time.

    :param dataset_or_model: DataFrame to iterate over or a model from which to use the dataset
    :param colname group: Name of the column to use for grouping
    :param name_pattern: Name to use for generated datasets. A number starting from 1 will
        be put in the placeholder.
    :returns: Tuple of DataFrame and the omitted group
    """

    def __init__(self, dataset_or_model, group, name_pattern='omitted_{}'):
        df = self._retrieve_dataset(dataset_or_model)
        self._unique_groups = df[group].unique()
        if len(self._unique_groups) == 1:
            raise ValueError("Cannot create an Omit iterator as the number of unique groups is 1.")
        self._df = df
        self._group = group
        super().__init__(len(self._unique_groups), name_pattern=name_pattern)

    def __next__(self):
        self._check_exhausted()
        df = self._df
        next_group = self._unique_groups[self._next - 1]
        new_df = df[df[self._group] != next_group]
        self._prepare_next(new_df)
        return self._combine_dataset(new_df), next_group


class Resample(DatasetIterator):
    """Iterate over resamples of a dataset.

    The dataset will be grouped on the group column then groups will be selected
    randomly with or without replacement to form a new dataset.
    The groups will be renumbered from 1 and upwards to keep them separated in the new
    dataset.

    Stratification will make sure that

    :param DataFrame df: DataFrame to iterate over
    :param colname group: Name of column to group by
    :param Int resamples: Number of resamples (iterations) to make
    :param colname stratify: Name of column to use for stratification.
        The values in the stratification column must be equal within a group so that the group
        can be uniquely determined. A ValueError exception will be raised otherwise.
    :param Int sample_size: The number of groups that should be sampled. The default is
        the number of groups. If using stratification the default is to sample using the
        proportion of the stratas in the dataset. A dictionary of specific sample sizes
        for each strata can also be supplied.
    :param bool replace: A boolean controlling whether sampling should be done with or
        without replacement
    :param name_pattern: Name to use for generated datasets. A number starting from 1 will
        be put in the placeholder.

    :returns: A tuple of a resampled DataFrame and a list of resampled groups in order
    """

    def __init__(
        self,
        dataset_or_model,
        group,
        resamples=1,
        stratify=None,
        sample_size=None,
        replace=False,
        name_pattern='resample_{}',
        name=None,
    ):
        df = self._retrieve_dataset(dataset_or_model)
        unique_groups = df[group].unique()
        numgroups = len(unique_groups)

        if sample_size is None:
            sample_size = numgroups

        if stratify:
            # Default is to use proportions in dataset
            stratas = df.groupby(stratify)[group].unique()
            have_mult_sample_sizes = isinstance(sample_size, collections.abc.Mapping)
            if not have_mult_sample_sizes:
                non_rounded_sample_sizes = stratas.apply(
                    lambda x: (len(x) / numgroups) * sample_size
                )
                rounded_sample_sizes = pharmpy.math.round_and_keep_sum(
                    non_rounded_sample_sizes, sample_size
                )
                sample_size_dict = dict(rounded_sample_sizes)  # strata: numsamples
            else:
                sample_size_dict = sample_size

            stratas = dict(stratas)  # strata: list of groups
        else:
            sample_size_dict = {1: sample_size}
            stratas = {1: unique_groups}

        # Check that we will not run out of samples without replacement.
        if not replace:
            for strata in sample_size_dict:
                if sample_size_dict[strata] > len(stratas[strata]):
                    if stratify:
                        raise ValueError(
                            f'The sample size ({sample_size_dict[strata]}) for strata {strata} is '
                            f'larger than the number of groups ({len(stratas[strata])}) in that '
                            f'strata which is impossible with replacement.'
                        )
                    else:
                        raise ValueError(
                            f'The sample size ({sample_size_dict[strata]}) is larger than the '
                            f'number of groups ({len(stratas[strata])}) which is impossible with '
                            f'replacement.'
                        )

        self._df = df
        self._group = group
        self._replace = replace
        self._stratas = stratas
        self._sample_size_dict = sample_size_dict
        if resamples > 1 and name:
            warnings.warn(
                f'One name was provided despite having multiple resamples, falling back to '
                f'name pattern: {name_pattern}'
            )
            self._name = None
        else:
            self._name = name
        super().__init__(resamples, name_pattern=name_pattern)

    def __next__(self):
        self._check_exhausted()

        random_groups = []
        for strata in self._sample_size_dict:
            random_groups += np.random.choice(
                self._stratas[strata], size=self._sample_size_dict[strata], replace=self._replace
            ).tolist()

        new_df = pd.DataFrame()
        # Build the dataset given the random_groups list
        for grp_id, new_grp in zip(random_groups, range(1, len(random_groups) + 1)):
            sub = self._df.loc[self._df[self._group] == grp_id].copy()
            sub[self._group] = new_grp
            new_df = new_df.append(sub)
        new_df.reset_index(inplace=True, drop=True)
        if self._name:
            new_df.name = self._name
        else:
            self._prepare_next(new_df)

        return self._combine_dataset(new_df), random_groups
